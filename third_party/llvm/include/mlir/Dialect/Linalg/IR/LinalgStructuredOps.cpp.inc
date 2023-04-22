/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::linalg::BatchMatmulOp,
::mlir::linalg::BatchMatmulTransposeBOp,
::mlir::linalg::BatchMatvecOp,
::mlir::linalg::BatchReduceMatmulOp,
::mlir::linalg::BroadcastOp,
::mlir::linalg::Conv1DNcwFcwOp,
::mlir::linalg::Conv1DNwcWcfOp,
::mlir::linalg::Conv1DOp,
::mlir::linalg::Conv2DNchwFchwOp,
::mlir::linalg::Conv2DNgchwFgchwOp,
::mlir::linalg::Conv2DNhwcFhwcOp,
::mlir::linalg::Conv2DNhwcHwcfOp,
::mlir::linalg::Conv2DNhwcHwcfQOp,
::mlir::linalg::Conv2DOp,
::mlir::linalg::Conv3DNdhwcDhwcfOp,
::mlir::linalg::Conv3DNdhwcDhwcfQOp,
::mlir::linalg::Conv3DOp,
::mlir::linalg::CopyOp,
::mlir::linalg::DepthwiseConv1DNwcWcOp,
::mlir::linalg::DepthwiseConv1DNwcWcmOp,
::mlir::linalg::DepthwiseConv2DNchwChwOp,
::mlir::linalg::DepthwiseConv2DNhwcHwcOp,
::mlir::linalg::DepthwiseConv2DNhwcHwcQOp,
::mlir::linalg::DepthwiseConv2DNhwcHwcmOp,
::mlir::linalg::DepthwiseConv2DNhwcHwcmQOp,
::mlir::linalg::DepthwiseConv3DNdhwcDhwcOp,
::mlir::linalg::DepthwiseConv3DNdhwcDhwcmOp,
::mlir::linalg::DotOp,
::mlir::linalg::ElemwiseBinaryOp,
::mlir::linalg::ElemwiseUnaryOp,
::mlir::linalg::FillOp,
::mlir::linalg::FillRng2DOp,
::mlir::linalg::GenericOp,
::mlir::linalg::MapOp,
::mlir::linalg::MatmulOp,
::mlir::linalg::MatmulTransposeBOp,
::mlir::linalg::MatmulUnsignedOp,
::mlir::linalg::MatvecOp,
::mlir::linalg::Mmt4DOp,
::mlir::linalg::PoolingNchwMaxOp,
::mlir::linalg::PoolingNchwSumOp,
::mlir::linalg::PoolingNcwMaxOp,
::mlir::linalg::PoolingNcwSumOp,
::mlir::linalg::PoolingNdhwcMaxOp,
::mlir::linalg::PoolingNdhwcMinOp,
::mlir::linalg::PoolingNdhwcSumOp,
::mlir::linalg::PoolingNhwcMaxOp,
::mlir::linalg::PoolingNhwcMaxUnsignedOp,
::mlir::linalg::PoolingNhwcMinOp,
::mlir::linalg::PoolingNhwcMinUnsignedOp,
::mlir::linalg::PoolingNhwcSumOp,
::mlir::linalg::PoolingNwcMaxOp,
::mlir::linalg::PoolingNwcMaxUnsignedOp,
::mlir::linalg::PoolingNwcMinOp,
::mlir::linalg::PoolingNwcMinUnsignedOp,
::mlir::linalg::PoolingNwcSumOp,
::mlir::linalg::QuantizedBatchMatmulOp,
::mlir::linalg::QuantizedMatmulOp,
::mlir::linalg::ReduceOp,
::mlir::linalg::TransposeOp,
::mlir::linalg::VecmatOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace linalg {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LinalgStructuredOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LinalgStructuredOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::ShapedType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be shaped of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LinalgStructuredOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::RankedTensorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be ranked tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LinalgStructuredOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType()))) || (((type.isa<::mlir::RankedTensorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be memref of any type values or ranked tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LinalgStructuredOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::DenseI64ArrayAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: i64 dense array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::DenseIntElementsAttr>())) && ((attr.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((attr.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1}))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::DenseIntElementsAttr>())) && ((attr.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((attr.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::DenseIntElementsAttr>())) && ((attr.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((attr.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3}))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::linalg::TypeFnAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::linalg::BinaryFnAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::linalg::UnaryFnAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps7(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::AffineMapAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: AffineMap array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps8(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::linalg::IteratorTypeAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: Iterator type should be an enum.";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps9(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::StringAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps10(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::DenseI64ArrayAttr>())) && (((llvm::is_sorted(attr.cast<DenseI64ArrayAttr>().asArrayRef()))) && (([](::llvm::ArrayRef<int64_t> a) {
return std::adjacent_find(std::begin(a), std::end(a)) == std::end(a);
}(attr.cast<DenseI64ArrayAttr>().asArrayRef())))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: i64 dense array attribute should be in increasing order";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_LinalgStructuredOps0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_LinalgStructuredOps1(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItems(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with 1 blocks";
  }
  return ::mlir::success();
}
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::BatchMatmulOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
BatchMatmulOpGenericAdaptorBase::BatchMatmulOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.batch_matmul", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> BatchMatmulOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BatchMatmulOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr BatchMatmulOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &BatchMatmulOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange BatchMatmulOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
BatchMatmulOpAdaptor::BatchMatmulOpAdaptor(BatchMatmulOp op) : BatchMatmulOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult BatchMatmulOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.batch_matmul' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == BatchMatmulOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.batch_matmul' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> BatchMatmulOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range BatchMatmulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range BatchMatmulOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range BatchMatmulOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange BatchMatmulOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange BatchMatmulOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> BatchMatmulOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range BatchMatmulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range BatchMatmulOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &BatchMatmulOp::getRegion() {
  return (*this)->getRegion(0);
}

void BatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, BatchMatmulOp::getRegionBuilder());
      
}

void BatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, BatchMatmulOp::getRegionBuilder());
      
}

void BatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult BatchMatmulOp::verifyInvariantsImpl() {
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
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult BatchMatmulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::BatchMatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::BatchMatmulTransposeBOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
BatchMatmulTransposeBOpGenericAdaptorBase::BatchMatmulTransposeBOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.batch_matmul_transpose_b", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> BatchMatmulTransposeBOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BatchMatmulTransposeBOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr BatchMatmulTransposeBOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &BatchMatmulTransposeBOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange BatchMatmulTransposeBOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
BatchMatmulTransposeBOpAdaptor::BatchMatmulTransposeBOpAdaptor(BatchMatmulTransposeBOp op) : BatchMatmulTransposeBOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult BatchMatmulTransposeBOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.batch_matmul_transpose_b' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == BatchMatmulTransposeBOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.batch_matmul_transpose_b' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> BatchMatmulTransposeBOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range BatchMatmulTransposeBOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range BatchMatmulTransposeBOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range BatchMatmulTransposeBOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange BatchMatmulTransposeBOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange BatchMatmulTransposeBOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> BatchMatmulTransposeBOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range BatchMatmulTransposeBOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range BatchMatmulTransposeBOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &BatchMatmulTransposeBOp::getRegion() {
  return (*this)->getRegion(0);
}

void BatchMatmulTransposeBOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, BatchMatmulTransposeBOp::getRegionBuilder());
      
}

void BatchMatmulTransposeBOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, BatchMatmulTransposeBOp::getRegionBuilder());
      
}

void BatchMatmulTransposeBOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult BatchMatmulTransposeBOp::verifyInvariantsImpl() {
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
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult BatchMatmulTransposeBOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::BatchMatmulTransposeBOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::BatchMatvecOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
BatchMatvecOpGenericAdaptorBase::BatchMatvecOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.batch_matvec", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> BatchMatvecOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BatchMatvecOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr BatchMatvecOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &BatchMatvecOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange BatchMatvecOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
BatchMatvecOpAdaptor::BatchMatvecOpAdaptor(BatchMatvecOp op) : BatchMatvecOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult BatchMatvecOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.batch_matvec' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == BatchMatvecOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.batch_matvec' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> BatchMatvecOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range BatchMatvecOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range BatchMatvecOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range BatchMatvecOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange BatchMatvecOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange BatchMatvecOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> BatchMatvecOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range BatchMatvecOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range BatchMatvecOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &BatchMatvecOp::getRegion() {
  return (*this)->getRegion(0);
}

void BatchMatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, BatchMatvecOp::getRegionBuilder());
      
}

void BatchMatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, BatchMatvecOp::getRegionBuilder());
      
}

void BatchMatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult BatchMatvecOp::verifyInvariantsImpl() {
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
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult BatchMatvecOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::BatchMatvecOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::BatchReduceMatmulOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
BatchReduceMatmulOpGenericAdaptorBase::BatchReduceMatmulOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.batch_reduce_matmul", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> BatchReduceMatmulOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BatchReduceMatmulOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr BatchReduceMatmulOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &BatchReduceMatmulOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange BatchReduceMatmulOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
BatchReduceMatmulOpAdaptor::BatchReduceMatmulOpAdaptor(BatchReduceMatmulOp op) : BatchReduceMatmulOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult BatchReduceMatmulOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.batch_reduce_matmul' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == BatchReduceMatmulOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.batch_reduce_matmul' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> BatchReduceMatmulOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range BatchReduceMatmulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range BatchReduceMatmulOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range BatchReduceMatmulOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange BatchReduceMatmulOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange BatchReduceMatmulOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> BatchReduceMatmulOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range BatchReduceMatmulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range BatchReduceMatmulOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &BatchReduceMatmulOp::getRegion() {
  return (*this)->getRegion(0);
}

void BatchReduceMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, BatchReduceMatmulOp::getRegionBuilder());
      
}

void BatchReduceMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, BatchReduceMatmulOp::getRegionBuilder());
      
}

void BatchReduceMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult BatchReduceMatmulOp::verifyInvariantsImpl() {
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
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult BatchReduceMatmulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::BatchReduceMatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::BroadcastOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
BroadcastOpGenericAdaptorBase::BroadcastOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.broadcast", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> BroadcastOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr BroadcastOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr BroadcastOpGenericAdaptorBase::getDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastOp::getDimensionsAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> BroadcastOpGenericAdaptorBase::getDimensions() {
  auto attr = getDimensionsAttr();
  return attr;
}

::mlir::Region &BroadcastOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange BroadcastOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
BroadcastOpAdaptor::BroadcastOpAdaptor(BroadcastOp op) : BroadcastOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult BroadcastOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.broadcast' op ""requires attribute 'dimensions'");
    if (namedAttrIt->getName() == BroadcastOp::getDimensionsAttrName(*odsOpName)) {
      tblgen_dimensions = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_dimensions && !((tblgen_dimensions.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'linalg.broadcast' op ""attribute 'dimensions' failed to satisfy constraint: i64 dense array attribute");
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

::mlir::TypedValue<::mlir::ShapedType> BroadcastOp::getInput() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::ShapedType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::ShapedType> BroadcastOp::getInit() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::ShapedType>>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange BroadcastOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastOp::getInitMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range BroadcastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range BroadcastOp::getResult() {
  return getODSResults(0);
}

::mlir::Region &BroadcastOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseI64ArrayAttr BroadcastOp::getDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getDimensionsAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> BroadcastOp::getDimensions() {
  auto attr = getDimensionsAttr();
  return attr;
}

void BroadcastOp::setDimensionsAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getDimensionsAttrName(), attr);
}

void BroadcastOp::setDimensions(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getDimensionsAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

::mlir::LogicalResult BroadcastOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'dimensions'");
    if (namedAttrIt->getName() == getDimensionsAttrName()) {
      tblgen_dimensions = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps0(*this, tblgen_dimensions, "dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::BroadcastOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv1DNcwFcwOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
Conv1DNcwFcwOpGenericAdaptorBase::Conv1DNcwFcwOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_1d_ncw_fcw", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> Conv1DNcwFcwOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv1DNcwFcwOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr Conv1DNcwFcwOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv1DNcwFcwOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv1DNcwFcwOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &Conv1DNcwFcwOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange Conv1DNcwFcwOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
Conv1DNcwFcwOpAdaptor::Conv1DNcwFcwOpAdaptor(Conv1DNcwFcwOp op) : Conv1DNcwFcwOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult Conv1DNcwFcwOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.conv_1d_ncw_fcw' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == Conv1DNcwFcwOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == Conv1DNcwFcwOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == Conv1DNcwFcwOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.conv_1d_ncw_fcw' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.conv_1d_ncw_fcw' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.conv_1d_ncw_fcw' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv1DNcwFcwOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range Conv1DNcwFcwOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv1DNcwFcwOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv1DNcwFcwOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv1DNcwFcwOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv1DNcwFcwOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv1DNcwFcwOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Conv1DNcwFcwOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Conv1DNcwFcwOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &Conv1DNcwFcwOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

void Conv1DNcwFcwOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void Conv1DNcwFcwOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute Conv1DNcwFcwOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute Conv1DNcwFcwOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void Conv1DNcwFcwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, Conv1DNcwFcwOp::getRegionBuilder());
      
}

void Conv1DNcwFcwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, Conv1DNcwFcwOp::getRegionBuilder());
      
}

void Conv1DNcwFcwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void Conv1DNcwFcwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, Conv1DNcwFcwOp::getRegionBuilder());
  
}

::mlir::LogicalResult Conv1DNcwFcwOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult Conv1DNcwFcwOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv1DNcwFcwOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv1DNwcWcfOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
Conv1DNwcWcfOpGenericAdaptorBase::Conv1DNwcWcfOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_1d_nwc_wcf", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> Conv1DNwcWcfOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv1DNwcWcfOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr Conv1DNwcWcfOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv1DNwcWcfOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv1DNwcWcfOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &Conv1DNwcWcfOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange Conv1DNwcWcfOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
Conv1DNwcWcfOpAdaptor::Conv1DNwcWcfOpAdaptor(Conv1DNwcWcfOp op) : Conv1DNwcWcfOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult Conv1DNwcWcfOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.conv_1d_nwc_wcf' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == Conv1DNwcWcfOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == Conv1DNwcWcfOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == Conv1DNwcWcfOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.conv_1d_nwc_wcf' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.conv_1d_nwc_wcf' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.conv_1d_nwc_wcf' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv1DNwcWcfOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range Conv1DNwcWcfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv1DNwcWcfOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv1DNwcWcfOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv1DNwcWcfOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv1DNwcWcfOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv1DNwcWcfOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Conv1DNwcWcfOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Conv1DNwcWcfOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &Conv1DNwcWcfOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

void Conv1DNwcWcfOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void Conv1DNwcWcfOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute Conv1DNwcWcfOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute Conv1DNwcWcfOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void Conv1DNwcWcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, Conv1DNwcWcfOp::getRegionBuilder());
      
}

void Conv1DNwcWcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, Conv1DNwcWcfOp::getRegionBuilder());
      
}

void Conv1DNwcWcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void Conv1DNwcWcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, Conv1DNwcWcfOp::getRegionBuilder());
  
}

::mlir::LogicalResult Conv1DNwcWcfOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult Conv1DNwcWcfOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv1DNwcWcfOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv1DOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
Conv1DOpGenericAdaptorBase::Conv1DOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_1d", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> Conv1DOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv1DOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr Conv1DOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &Conv1DOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange Conv1DOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
Conv1DOpAdaptor::Conv1DOpAdaptor(Conv1DOp op) : Conv1DOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult Conv1DOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.conv_1d' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == Conv1DOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.conv_1d' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv1DOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range Conv1DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv1DOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv1DOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv1DOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv1DOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv1DOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Conv1DOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Conv1DOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &Conv1DOp::getRegion() {
  return (*this)->getRegion(0);
}

void Conv1DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, Conv1DOp::getRegionBuilder());
      
}

void Conv1DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, Conv1DOp::getRegionBuilder());
      
}

void Conv1DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult Conv1DOp::verifyInvariantsImpl() {
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
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult Conv1DOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv1DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DNchwFchwOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
Conv2DNchwFchwOpGenericAdaptorBase::Conv2DNchwFchwOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_2d_nchw_fchw", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> Conv2DNchwFchwOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv2DNchwFchwOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr Conv2DNchwFchwOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv2DNchwFchwOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv2DNchwFchwOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &Conv2DNchwFchwOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange Conv2DNchwFchwOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
Conv2DNchwFchwOpAdaptor::Conv2DNchwFchwOpAdaptor(Conv2DNchwFchwOp op) : Conv2DNchwFchwOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult Conv2DNchwFchwOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.conv_2d_nchw_fchw' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == Conv2DNchwFchwOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == Conv2DNchwFchwOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == Conv2DNchwFchwOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.conv_2d_nchw_fchw' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.conv_2d_nchw_fchw' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.conv_2d_nchw_fchw' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv2DNchwFchwOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range Conv2DNchwFchwOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv2DNchwFchwOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv2DNchwFchwOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv2DNchwFchwOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNchwFchwOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv2DNchwFchwOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Conv2DNchwFchwOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Conv2DNchwFchwOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &Conv2DNchwFchwOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void Conv2DNchwFchwOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void Conv2DNchwFchwOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute Conv2DNchwFchwOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute Conv2DNchwFchwOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void Conv2DNchwFchwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, Conv2DNchwFchwOp::getRegionBuilder());
      
}

void Conv2DNchwFchwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, Conv2DNchwFchwOp::getRegionBuilder());
      
}

void Conv2DNchwFchwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void Conv2DNchwFchwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, Conv2DNchwFchwOp::getRegionBuilder());
  
}

::mlir::LogicalResult Conv2DNchwFchwOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult Conv2DNchwFchwOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DNchwFchwOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DNgchwFgchwOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
Conv2DNgchwFgchwOpGenericAdaptorBase::Conv2DNgchwFgchwOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_2d_ngchw_fgchw", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> Conv2DNgchwFgchwOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv2DNgchwFgchwOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr Conv2DNgchwFgchwOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv2DNgchwFgchwOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv2DNgchwFgchwOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &Conv2DNgchwFgchwOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange Conv2DNgchwFgchwOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
Conv2DNgchwFgchwOpAdaptor::Conv2DNgchwFgchwOpAdaptor(Conv2DNgchwFgchwOp op) : Conv2DNgchwFgchwOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult Conv2DNgchwFgchwOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.conv_2d_ngchw_fgchw' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == Conv2DNgchwFgchwOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == Conv2DNgchwFgchwOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == Conv2DNgchwFgchwOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.conv_2d_ngchw_fgchw' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.conv_2d_ngchw_fgchw' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.conv_2d_ngchw_fgchw' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv2DNgchwFgchwOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range Conv2DNgchwFgchwOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv2DNgchwFgchwOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv2DNgchwFgchwOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv2DNgchwFgchwOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNgchwFgchwOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv2DNgchwFgchwOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Conv2DNgchwFgchwOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Conv2DNgchwFgchwOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &Conv2DNgchwFgchwOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void Conv2DNgchwFgchwOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void Conv2DNgchwFgchwOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute Conv2DNgchwFgchwOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute Conv2DNgchwFgchwOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void Conv2DNgchwFgchwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, Conv2DNgchwFgchwOp::getRegionBuilder());
      
}

void Conv2DNgchwFgchwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, Conv2DNgchwFgchwOp::getRegionBuilder());
      
}

void Conv2DNgchwFgchwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void Conv2DNgchwFgchwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, Conv2DNgchwFgchwOp::getRegionBuilder());
  
}

::mlir::LogicalResult Conv2DNgchwFgchwOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult Conv2DNgchwFgchwOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DNgchwFgchwOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DNhwcFhwcOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
Conv2DNhwcFhwcOpGenericAdaptorBase::Conv2DNhwcFhwcOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_2d_nhwc_fhwc", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> Conv2DNhwcFhwcOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv2DNhwcFhwcOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr Conv2DNhwcFhwcOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv2DNhwcFhwcOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv2DNhwcFhwcOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &Conv2DNhwcFhwcOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange Conv2DNhwcFhwcOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
Conv2DNhwcFhwcOpAdaptor::Conv2DNhwcFhwcOpAdaptor(Conv2DNhwcFhwcOp op) : Conv2DNhwcFhwcOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult Conv2DNhwcFhwcOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.conv_2d_nhwc_fhwc' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == Conv2DNhwcFhwcOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == Conv2DNhwcFhwcOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == Conv2DNhwcFhwcOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.conv_2d_nhwc_fhwc' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.conv_2d_nhwc_fhwc' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.conv_2d_nhwc_fhwc' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv2DNhwcFhwcOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range Conv2DNhwcFhwcOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv2DNhwcFhwcOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv2DNhwcFhwcOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv2DNhwcFhwcOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNhwcFhwcOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv2DNhwcFhwcOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Conv2DNhwcFhwcOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Conv2DNhwcFhwcOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &Conv2DNhwcFhwcOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void Conv2DNhwcFhwcOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void Conv2DNhwcFhwcOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute Conv2DNhwcFhwcOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute Conv2DNhwcFhwcOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void Conv2DNhwcFhwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, Conv2DNhwcFhwcOp::getRegionBuilder());
      
}

void Conv2DNhwcFhwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, Conv2DNhwcFhwcOp::getRegionBuilder());
      
}

void Conv2DNhwcFhwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void Conv2DNhwcFhwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, Conv2DNhwcFhwcOp::getRegionBuilder());
  
}

::mlir::LogicalResult Conv2DNhwcFhwcOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult Conv2DNhwcFhwcOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DNhwcFhwcOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DNhwcHwcfOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
Conv2DNhwcHwcfOpGenericAdaptorBase::Conv2DNhwcHwcfOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_2d_nhwc_hwcf", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> Conv2DNhwcHwcfOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv2DNhwcHwcfOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr Conv2DNhwcHwcfOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv2DNhwcHwcfOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv2DNhwcHwcfOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &Conv2DNhwcHwcfOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange Conv2DNhwcHwcfOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
Conv2DNhwcHwcfOpAdaptor::Conv2DNhwcHwcfOpAdaptor(Conv2DNhwcHwcfOp op) : Conv2DNhwcHwcfOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult Conv2DNhwcHwcfOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.conv_2d_nhwc_hwcf' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == Conv2DNhwcHwcfOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == Conv2DNhwcHwcfOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == Conv2DNhwcHwcfOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.conv_2d_nhwc_hwcf' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.conv_2d_nhwc_hwcf' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.conv_2d_nhwc_hwcf' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv2DNhwcHwcfOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range Conv2DNhwcHwcfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv2DNhwcHwcfOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv2DNhwcHwcfOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv2DNhwcHwcfOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNhwcHwcfOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv2DNhwcHwcfOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Conv2DNhwcHwcfOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Conv2DNhwcHwcfOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &Conv2DNhwcHwcfOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void Conv2DNhwcHwcfOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void Conv2DNhwcHwcfOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute Conv2DNhwcHwcfOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute Conv2DNhwcHwcfOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void Conv2DNhwcHwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, Conv2DNhwcHwcfOp::getRegionBuilder());
      
}

void Conv2DNhwcHwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, Conv2DNhwcHwcfOp::getRegionBuilder());
      
}

void Conv2DNhwcHwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void Conv2DNhwcHwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, Conv2DNhwcHwcfOp::getRegionBuilder());
  
}

::mlir::LogicalResult Conv2DNhwcHwcfOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult Conv2DNhwcHwcfOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DNhwcHwcfOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DNhwcHwcfQOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
Conv2DNhwcHwcfQOpGenericAdaptorBase::Conv2DNhwcHwcfQOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_2d_nhwc_hwcf_q", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> Conv2DNhwcHwcfQOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv2DNhwcHwcfQOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr Conv2DNhwcHwcfQOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv2DNhwcHwcfQOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv2DNhwcHwcfQOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &Conv2DNhwcHwcfQOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange Conv2DNhwcHwcfQOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
Conv2DNhwcHwcfQOpAdaptor::Conv2DNhwcHwcfQOpAdaptor(Conv2DNhwcHwcfQOp op) : Conv2DNhwcHwcfQOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult Conv2DNhwcHwcfQOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.conv_2d_nhwc_hwcf_q' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == Conv2DNhwcHwcfQOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == Conv2DNhwcHwcfQOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == Conv2DNhwcHwcfQOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.conv_2d_nhwc_hwcf_q' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.conv_2d_nhwc_hwcf_q' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.conv_2d_nhwc_hwcf_q' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv2DNhwcHwcfQOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range Conv2DNhwcHwcfQOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv2DNhwcHwcfQOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv2DNhwcHwcfQOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv2DNhwcHwcfQOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNhwcHwcfQOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv2DNhwcHwcfQOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Conv2DNhwcHwcfQOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Conv2DNhwcHwcfQOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &Conv2DNhwcHwcfQOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void Conv2DNhwcHwcfQOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void Conv2DNhwcHwcfQOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute Conv2DNhwcHwcfQOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute Conv2DNhwcHwcfQOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void Conv2DNhwcHwcfQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, Conv2DNhwcHwcfQOp::getRegionBuilder());
      
}

void Conv2DNhwcHwcfQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, Conv2DNhwcHwcfQOp::getRegionBuilder());
      
}

void Conv2DNhwcHwcfQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void Conv2DNhwcHwcfQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, Conv2DNhwcHwcfQOp::getRegionBuilder());
  
}

::mlir::LogicalResult Conv2DNhwcHwcfQOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult Conv2DNhwcHwcfQOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DNhwcHwcfQOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
Conv2DOpGenericAdaptorBase::Conv2DOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_2d", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> Conv2DOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv2DOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr Conv2DOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &Conv2DOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange Conv2DOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
Conv2DOpAdaptor::Conv2DOpAdaptor(Conv2DOp op) : Conv2DOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult Conv2DOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.conv_2d' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == Conv2DOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.conv_2d' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv2DOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range Conv2DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv2DOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv2DOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv2DOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv2DOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Conv2DOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Conv2DOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &Conv2DOp::getRegion() {
  return (*this)->getRegion(0);
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, Conv2DOp::getRegionBuilder());
      
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, Conv2DOp::getRegionBuilder());
      
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult Conv2DOp::verifyInvariantsImpl() {
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
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult Conv2DOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv3DNdhwcDhwcfOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
Conv3DNdhwcDhwcfOpGenericAdaptorBase::Conv3DNdhwcDhwcfOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_3d_ndhwc_dhwcf", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> Conv3DNdhwcDhwcfOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv3DNdhwcDhwcfOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr Conv3DNdhwcDhwcfOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv3DNdhwcDhwcfOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv3DNdhwcDhwcfOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &Conv3DNdhwcDhwcfOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange Conv3DNdhwcDhwcfOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
Conv3DNdhwcDhwcfOpAdaptor::Conv3DNdhwcDhwcfOpAdaptor(Conv3DNdhwcDhwcfOp op) : Conv3DNdhwcDhwcfOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult Conv3DNdhwcDhwcfOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.conv_3d_ndhwc_dhwcf' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == Conv3DNdhwcDhwcfOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == Conv3DNdhwcDhwcfOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == Conv3DNdhwcDhwcfOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.conv_3d_ndhwc_dhwcf' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
    return emitError(loc, "'linalg.conv_3d_ndhwc_dhwcf' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
    return emitError(loc, "'linalg.conv_3d_ndhwc_dhwcf' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv3DNdhwcDhwcfOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range Conv3DNdhwcDhwcfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv3DNdhwcDhwcfOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv3DNdhwcDhwcfOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv3DNdhwcDhwcfOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv3DNdhwcDhwcfOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv3DNdhwcDhwcfOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Conv3DNdhwcDhwcfOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Conv3DNdhwcDhwcfOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &Conv3DNdhwcDhwcfOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void Conv3DNdhwcDhwcfOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void Conv3DNdhwcDhwcfOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute Conv3DNdhwcDhwcfOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute Conv3DNdhwcDhwcfOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void Conv3DNdhwcDhwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, Conv3DNdhwcDhwcfOp::getRegionBuilder());
      
}

void Conv3DNdhwcDhwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, Conv3DNdhwcDhwcfOp::getRegionBuilder());
      
}

void Conv3DNdhwcDhwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void Conv3DNdhwcDhwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, Conv3DNdhwcDhwcfOp::getRegionBuilder());
  
}

::mlir::LogicalResult Conv3DNdhwcDhwcfOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps3(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps3(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult Conv3DNdhwcDhwcfOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv3DNdhwcDhwcfOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv3DNdhwcDhwcfQOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
Conv3DNdhwcDhwcfQOpGenericAdaptorBase::Conv3DNdhwcDhwcfQOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_3d_ndhwc_dhwcf_q", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> Conv3DNdhwcDhwcfQOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv3DNdhwcDhwcfQOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr Conv3DNdhwcDhwcfQOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfQOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv3DNdhwcDhwcfQOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfQOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfQOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv3DNdhwcDhwcfQOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfQOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &Conv3DNdhwcDhwcfQOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange Conv3DNdhwcDhwcfQOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
Conv3DNdhwcDhwcfQOpAdaptor::Conv3DNdhwcDhwcfQOpAdaptor(Conv3DNdhwcDhwcfQOp op) : Conv3DNdhwcDhwcfQOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult Conv3DNdhwcDhwcfQOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.conv_3d_ndhwc_dhwcf_q' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == Conv3DNdhwcDhwcfQOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == Conv3DNdhwcDhwcfQOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == Conv3DNdhwcDhwcfQOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.conv_3d_ndhwc_dhwcf_q' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
    return emitError(loc, "'linalg.conv_3d_ndhwc_dhwcf_q' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
    return emitError(loc, "'linalg.conv_3d_ndhwc_dhwcf_q' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv3DNdhwcDhwcfQOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range Conv3DNdhwcDhwcfQOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv3DNdhwcDhwcfQOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv3DNdhwcDhwcfQOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv3DNdhwcDhwcfQOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv3DNdhwcDhwcfQOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv3DNdhwcDhwcfQOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Conv3DNdhwcDhwcfQOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Conv3DNdhwcDhwcfQOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &Conv3DNdhwcDhwcfQOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfQOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfQOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfQOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfQOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void Conv3DNdhwcDhwcfQOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void Conv3DNdhwcDhwcfQOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute Conv3DNdhwcDhwcfQOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute Conv3DNdhwcDhwcfQOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void Conv3DNdhwcDhwcfQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, Conv3DNdhwcDhwcfQOp::getRegionBuilder());
      
}

void Conv3DNdhwcDhwcfQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, Conv3DNdhwcDhwcfQOp::getRegionBuilder());
      
}

void Conv3DNdhwcDhwcfQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void Conv3DNdhwcDhwcfQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, Conv3DNdhwcDhwcfQOp::getRegionBuilder());
  
}

::mlir::LogicalResult Conv3DNdhwcDhwcfQOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps3(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps3(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult Conv3DNdhwcDhwcfQOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv3DNdhwcDhwcfQOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv3DOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
Conv3DOpGenericAdaptorBase::Conv3DOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_3d", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> Conv3DOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv3DOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr Conv3DOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &Conv3DOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange Conv3DOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
Conv3DOpAdaptor::Conv3DOpAdaptor(Conv3DOp op) : Conv3DOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult Conv3DOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.conv_3d' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == Conv3DOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.conv_3d' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv3DOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range Conv3DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv3DOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv3DOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv3DOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv3DOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv3DOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Conv3DOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Conv3DOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &Conv3DOp::getRegion() {
  return (*this)->getRegion(0);
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, Conv3DOp::getRegionBuilder());
      
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, Conv3DOp::getRegionBuilder());
      
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult Conv3DOp::verifyInvariantsImpl() {
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
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult Conv3DOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv3DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::CopyOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CopyOpGenericAdaptorBase::CopyOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.copy", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CopyOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CopyOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr CopyOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::linalg::TypeFnAttr CopyOpGenericAdaptorBase::getCastAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CopyOp::getCastAttrName(*odsOpName)).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
  if (!attr)
    attr = ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed);
  return attr;
}

::mlir::linalg::TypeFn CopyOpGenericAdaptorBase::getCast() {
  auto attr = getCastAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

::mlir::Region &CopyOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange CopyOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
CopyOpAdaptor::CopyOpAdaptor(CopyOp op) : CopyOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CopyOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_cast;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.copy' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == CopyOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == CopyOp::getCastAttrName(*odsOpName)) {
      tblgen_cast = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.copy' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_cast && !((tblgen_cast.isa<::mlir::linalg::TypeFnAttr>())))
    return emitError(loc, "'linalg.copy' op ""attribute 'cast' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CopyOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range CopyOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range CopyOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range CopyOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange CopyOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange CopyOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> CopyOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range CopyOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range CopyOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &CopyOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::linalg::TypeFnAttr CopyOp::getCastAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCastAttrName()).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
}

::mlir::linalg::TypeFn CopyOp::getCast() {
  auto attr = getCastAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

void CopyOp::setCastAttr(::mlir::linalg::TypeFnAttr attr) {
  (*this)->setAttr(getCastAttrName(), attr);
}

void CopyOp::setCast(::std::optional<::mlir::linalg::TypeFn> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getCastAttrName(), ::mlir::linalg::TypeFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getCastAttrName());
}

::mlir::Attribute CopyOp::removeCastAttr() {
  return (*this)->removeAttr(getCastAttrName());
}

void CopyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, CopyOp::getRegionBuilder());
      
}

void CopyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, CopyOp::getRegionBuilder());
      
}

void CopyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void CopyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute cast, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("cast", cast);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, CopyOp::getRegionBuilder());
  
}

::mlir::LogicalResult CopyOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_cast;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getCastAttrName()) {
      tblgen_cast = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps4(*this, tblgen_cast, "cast")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult CopyOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::CopyOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv1DNwcWcOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DepthwiseConv1DNwcWcOpGenericAdaptorBase::DepthwiseConv1DNwcWcOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.depthwise_conv_1d_nwc_wc", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DepthwiseConv1DNwcWcOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DepthwiseConv1DNwcWcOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr DepthwiseConv1DNwcWcOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv1DNwcWcOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv1DNwcWcOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &DepthwiseConv1DNwcWcOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange DepthwiseConv1DNwcWcOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
DepthwiseConv1DNwcWcOpAdaptor::DepthwiseConv1DNwcWcOpAdaptor(DepthwiseConv1DNwcWcOp op) : DepthwiseConv1DNwcWcOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult DepthwiseConv1DNwcWcOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.depthwise_conv_1d_nwc_wc' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == DepthwiseConv1DNwcWcOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == DepthwiseConv1DNwcWcOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == DepthwiseConv1DNwcWcOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.depthwise_conv_1d_nwc_wc' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.depthwise_conv_1d_nwc_wc' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.depthwise_conv_1d_nwc_wc' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DepthwiseConv1DNwcWcOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range DepthwiseConv1DNwcWcOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DepthwiseConv1DNwcWcOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv1DNwcWcOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv1DNwcWcOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv1DNwcWcOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DepthwiseConv1DNwcWcOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DepthwiseConv1DNwcWcOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DepthwiseConv1DNwcWcOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv1DNwcWcOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv1DNwcWcOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void DepthwiseConv1DNwcWcOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute DepthwiseConv1DNwcWcOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute DepthwiseConv1DNwcWcOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void DepthwiseConv1DNwcWcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, DepthwiseConv1DNwcWcOp::getRegionBuilder());
      
}

void DepthwiseConv1DNwcWcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, DepthwiseConv1DNwcWcOp::getRegionBuilder());
      
}

void DepthwiseConv1DNwcWcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void DepthwiseConv1DNwcWcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, DepthwiseConv1DNwcWcOp::getRegionBuilder());
  
}

::mlir::LogicalResult DepthwiseConv1DNwcWcOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult DepthwiseConv1DNwcWcOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv1DNwcWcOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv1DNwcWcmOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DepthwiseConv1DNwcWcmOpGenericAdaptorBase::DepthwiseConv1DNwcWcmOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.depthwise_conv_1d_nwc_wcm", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DepthwiseConv1DNwcWcmOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DepthwiseConv1DNwcWcmOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr DepthwiseConv1DNwcWcmOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv1DNwcWcmOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv1DNwcWcmOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &DepthwiseConv1DNwcWcmOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange DepthwiseConv1DNwcWcmOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
DepthwiseConv1DNwcWcmOpAdaptor::DepthwiseConv1DNwcWcmOpAdaptor(DepthwiseConv1DNwcWcmOp op) : DepthwiseConv1DNwcWcmOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult DepthwiseConv1DNwcWcmOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.depthwise_conv_1d_nwc_wcm' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == DepthwiseConv1DNwcWcmOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == DepthwiseConv1DNwcWcmOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == DepthwiseConv1DNwcWcmOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.depthwise_conv_1d_nwc_wcm' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.depthwise_conv_1d_nwc_wcm' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.depthwise_conv_1d_nwc_wcm' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DepthwiseConv1DNwcWcmOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range DepthwiseConv1DNwcWcmOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DepthwiseConv1DNwcWcmOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv1DNwcWcmOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv1DNwcWcmOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv1DNwcWcmOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DepthwiseConv1DNwcWcmOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DepthwiseConv1DNwcWcmOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DepthwiseConv1DNwcWcmOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv1DNwcWcmOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv1DNwcWcmOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void DepthwiseConv1DNwcWcmOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute DepthwiseConv1DNwcWcmOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute DepthwiseConv1DNwcWcmOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void DepthwiseConv1DNwcWcmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, DepthwiseConv1DNwcWcmOp::getRegionBuilder());
      
}

void DepthwiseConv1DNwcWcmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, DepthwiseConv1DNwcWcmOp::getRegionBuilder());
      
}

void DepthwiseConv1DNwcWcmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void DepthwiseConv1DNwcWcmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, DepthwiseConv1DNwcWcmOp::getRegionBuilder());
  
}

::mlir::LogicalResult DepthwiseConv1DNwcWcmOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult DepthwiseConv1DNwcWcmOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv1DNwcWcmOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNchwChwOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DepthwiseConv2DNchwChwOpGenericAdaptorBase::DepthwiseConv2DNchwChwOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.depthwise_conv_2d_nchw_chw", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DepthwiseConv2DNchwChwOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DepthwiseConv2DNchwChwOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr DepthwiseConv2DNchwChwOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv2DNchwChwOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv2DNchwChwOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &DepthwiseConv2DNchwChwOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange DepthwiseConv2DNchwChwOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
DepthwiseConv2DNchwChwOpAdaptor::DepthwiseConv2DNchwChwOpAdaptor(DepthwiseConv2DNchwChwOp op) : DepthwiseConv2DNchwChwOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult DepthwiseConv2DNchwChwOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.depthwise_conv_2d_nchw_chw' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == DepthwiseConv2DNchwChwOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == DepthwiseConv2DNchwChwOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == DepthwiseConv2DNchwChwOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.depthwise_conv_2d_nchw_chw' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.depthwise_conv_2d_nchw_chw' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.depthwise_conv_2d_nchw_chw' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DepthwiseConv2DNchwChwOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range DepthwiseConv2DNchwChwOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DepthwiseConv2DNchwChwOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv2DNchwChwOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv2DNchwChwOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNchwChwOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNchwChwOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DepthwiseConv2DNchwChwOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DepthwiseConv2DNchwChwOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv2DNchwChwOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv2DNchwChwOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void DepthwiseConv2DNchwChwOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute DepthwiseConv2DNchwChwOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute DepthwiseConv2DNchwChwOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void DepthwiseConv2DNchwChwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, DepthwiseConv2DNchwChwOp::getRegionBuilder());
      
}

void DepthwiseConv2DNchwChwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, DepthwiseConv2DNchwChwOp::getRegionBuilder());
      
}

void DepthwiseConv2DNchwChwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void DepthwiseConv2DNchwChwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, DepthwiseConv2DNchwChwOp::getRegionBuilder());
  
}

::mlir::LogicalResult DepthwiseConv2DNchwChwOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult DepthwiseConv2DNchwChwOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNchwChwOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNhwcHwcOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DepthwiseConv2DNhwcHwcOpGenericAdaptorBase::DepthwiseConv2DNhwcHwcOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.depthwise_conv_2d_nhwc_hwc", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DepthwiseConv2DNhwcHwcOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr DepthwiseConv2DNhwcHwcOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv2DNhwcHwcOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv2DNhwcHwcOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &DepthwiseConv2DNhwcHwcOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange DepthwiseConv2DNhwcHwcOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
DepthwiseConv2DNhwcHwcOpAdaptor::DepthwiseConv2DNhwcHwcOpAdaptor(DepthwiseConv2DNhwcHwcOp op) : DepthwiseConv2DNhwcHwcOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult DepthwiseConv2DNhwcHwcOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwc' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == DepthwiseConv2DNhwcHwcOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == DepthwiseConv2DNhwcHwcOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == DepthwiseConv2DNhwcHwcOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwc' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwc' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwc' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DepthwiseConv2DNhwcHwcOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DepthwiseConv2DNhwcHwcOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv2DNhwcHwcOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv2DNhwcHwcOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute DepthwiseConv2DNhwcHwcOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute DepthwiseConv2DNhwcHwcOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void DepthwiseConv2DNhwcHwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, DepthwiseConv2DNhwcHwcOp::getRegionBuilder());
      
}

void DepthwiseConv2DNhwcHwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, DepthwiseConv2DNhwcHwcOp::getRegionBuilder());
      
}

void DepthwiseConv2DNhwcHwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void DepthwiseConv2DNhwcHwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, DepthwiseConv2DNhwcHwcOp::getRegionBuilder());
  
}

::mlir::LogicalResult DepthwiseConv2DNhwcHwcOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult DepthwiseConv2DNhwcHwcOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNhwcHwcOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNhwcHwcQOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DepthwiseConv2DNhwcHwcQOpGenericAdaptorBase::DepthwiseConv2DNhwcHwcQOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.depthwise_conv_2d_nhwc_hwc_q", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcQOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DepthwiseConv2DNhwcHwcQOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr DepthwiseConv2DNhwcHwcQOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv2DNhwcHwcQOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv2DNhwcHwcQOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &DepthwiseConv2DNhwcHwcQOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange DepthwiseConv2DNhwcHwcQOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
DepthwiseConv2DNhwcHwcQOpAdaptor::DepthwiseConv2DNhwcHwcQOpAdaptor(DepthwiseConv2DNhwcHwcQOp op) : DepthwiseConv2DNhwcHwcQOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult DepthwiseConv2DNhwcHwcQOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwc_q' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == DepthwiseConv2DNhwcHwcQOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == DepthwiseConv2DNhwcHwcQOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == DepthwiseConv2DNhwcHwcQOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwc_q' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwc_q' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwc_q' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcQOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcQOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcQOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcQOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcQOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcQOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcQOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DepthwiseConv2DNhwcHwcQOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DepthwiseConv2DNhwcHwcQOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv2DNhwcHwcQOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv2DNhwcHwcQOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcQOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute DepthwiseConv2DNhwcHwcQOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute DepthwiseConv2DNhwcHwcQOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void DepthwiseConv2DNhwcHwcQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, DepthwiseConv2DNhwcHwcQOp::getRegionBuilder());
      
}

void DepthwiseConv2DNhwcHwcQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, DepthwiseConv2DNhwcHwcQOp::getRegionBuilder());
      
}

void DepthwiseConv2DNhwcHwcQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void DepthwiseConv2DNhwcHwcQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, DepthwiseConv2DNhwcHwcQOp::getRegionBuilder());
  
}

::mlir::LogicalResult DepthwiseConv2DNhwcHwcQOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult DepthwiseConv2DNhwcHwcQOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNhwcHwcQOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNhwcHwcmOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DepthwiseConv2DNhwcHwcmOpGenericAdaptorBase::DepthwiseConv2DNhwcHwcmOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.depthwise_conv_2d_nhwc_hwcm", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcmOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DepthwiseConv2DNhwcHwcmOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr DepthwiseConv2DNhwcHwcmOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv2DNhwcHwcmOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv2DNhwcHwcmOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &DepthwiseConv2DNhwcHwcmOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange DepthwiseConv2DNhwcHwcmOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
DepthwiseConv2DNhwcHwcmOpAdaptor::DepthwiseConv2DNhwcHwcmOpAdaptor(DepthwiseConv2DNhwcHwcmOp op) : DepthwiseConv2DNhwcHwcmOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult DepthwiseConv2DNhwcHwcmOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwcm' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == DepthwiseConv2DNhwcHwcmOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == DepthwiseConv2DNhwcHwcmOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == DepthwiseConv2DNhwcHwcmOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwcm' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwcm' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwcm' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcmOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcmOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcmOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcmOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcmOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcmOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcmOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DepthwiseConv2DNhwcHwcmOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DepthwiseConv2DNhwcHwcmOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv2DNhwcHwcmOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv2DNhwcHwcmOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcmOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute DepthwiseConv2DNhwcHwcmOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute DepthwiseConv2DNhwcHwcmOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void DepthwiseConv2DNhwcHwcmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, DepthwiseConv2DNhwcHwcmOp::getRegionBuilder());
      
}

void DepthwiseConv2DNhwcHwcmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, DepthwiseConv2DNhwcHwcmOp::getRegionBuilder());
      
}

void DepthwiseConv2DNhwcHwcmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void DepthwiseConv2DNhwcHwcmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, DepthwiseConv2DNhwcHwcmOp::getRegionBuilder());
  
}

::mlir::LogicalResult DepthwiseConv2DNhwcHwcmOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult DepthwiseConv2DNhwcHwcmOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNhwcHwcmOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNhwcHwcmQOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DepthwiseConv2DNhwcHwcmQOpGenericAdaptorBase::DepthwiseConv2DNhwcHwcmQOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.depthwise_conv_2d_nhwc_hwcm_q", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcmQOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DepthwiseConv2DNhwcHwcmQOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr DepthwiseConv2DNhwcHwcmQOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv2DNhwcHwcmQOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv2DNhwcHwcmQOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &DepthwiseConv2DNhwcHwcmQOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange DepthwiseConv2DNhwcHwcmQOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
DepthwiseConv2DNhwcHwcmQOpAdaptor::DepthwiseConv2DNhwcHwcmQOpAdaptor(DepthwiseConv2DNhwcHwcmQOp op) : DepthwiseConv2DNhwcHwcmQOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult DepthwiseConv2DNhwcHwcmQOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwcm_q' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == DepthwiseConv2DNhwcHwcmQOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == DepthwiseConv2DNhwcHwcmQOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == DepthwiseConv2DNhwcHwcmQOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwcm_q' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwcm_q' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwcm_q' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcmQOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcmQOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcmQOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcmQOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcmQOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcmQOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcmQOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DepthwiseConv2DNhwcHwcmQOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DepthwiseConv2DNhwcHwcmQOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv2DNhwcHwcmQOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv2DNhwcHwcmQOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcmQOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute DepthwiseConv2DNhwcHwcmQOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute DepthwiseConv2DNhwcHwcmQOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void DepthwiseConv2DNhwcHwcmQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, DepthwiseConv2DNhwcHwcmQOp::getRegionBuilder());
      
}

void DepthwiseConv2DNhwcHwcmQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, DepthwiseConv2DNhwcHwcmQOp::getRegionBuilder());
      
}

void DepthwiseConv2DNhwcHwcmQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void DepthwiseConv2DNhwcHwcmQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, DepthwiseConv2DNhwcHwcmQOp::getRegionBuilder());
  
}

::mlir::LogicalResult DepthwiseConv2DNhwcHwcmQOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult DepthwiseConv2DNhwcHwcmQOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNhwcHwcmQOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv3DNdhwcDhwcOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DepthwiseConv3DNdhwcDhwcOpGenericAdaptorBase::DepthwiseConv3DNdhwcDhwcOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.depthwise_conv_3d_ndhwc_dhwc", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DepthwiseConv3DNdhwcDhwcOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DepthwiseConv3DNdhwcDhwcOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr DepthwiseConv3DNdhwcDhwcOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv3DNdhwcDhwcOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv3DNdhwcDhwcOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &DepthwiseConv3DNdhwcDhwcOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange DepthwiseConv3DNdhwcDhwcOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
DepthwiseConv3DNdhwcDhwcOpAdaptor::DepthwiseConv3DNdhwcDhwcOpAdaptor(DepthwiseConv3DNdhwcDhwcOp op) : DepthwiseConv3DNdhwcDhwcOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult DepthwiseConv3DNdhwcDhwcOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.depthwise_conv_3d_ndhwc_dhwc' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == DepthwiseConv3DNdhwcDhwcOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == DepthwiseConv3DNdhwcDhwcOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == DepthwiseConv3DNdhwcDhwcOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.depthwise_conv_3d_ndhwc_dhwc' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
    return emitError(loc, "'linalg.depthwise_conv_3d_ndhwc_dhwc' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
    return emitError(loc, "'linalg.depthwise_conv_3d_ndhwc_dhwc' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DepthwiseConv3DNdhwcDhwcOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range DepthwiseConv3DNdhwcDhwcOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DepthwiseConv3DNdhwcDhwcOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv3DNdhwcDhwcOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv3DNdhwcDhwcOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv3DNdhwcDhwcOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DepthwiseConv3DNdhwcDhwcOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DepthwiseConv3DNdhwcDhwcOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DepthwiseConv3DNdhwcDhwcOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv3DNdhwcDhwcOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv3DNdhwcDhwcOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void DepthwiseConv3DNdhwcDhwcOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute DepthwiseConv3DNdhwcDhwcOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute DepthwiseConv3DNdhwcDhwcOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void DepthwiseConv3DNdhwcDhwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, DepthwiseConv3DNdhwcDhwcOp::getRegionBuilder());
      
}

void DepthwiseConv3DNdhwcDhwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, DepthwiseConv3DNdhwcDhwcOp::getRegionBuilder());
      
}

void DepthwiseConv3DNdhwcDhwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void DepthwiseConv3DNdhwcDhwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, DepthwiseConv3DNdhwcDhwcOp::getRegionBuilder());
  
}

::mlir::LogicalResult DepthwiseConv3DNdhwcDhwcOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps3(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps3(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult DepthwiseConv3DNdhwcDhwcOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv3DNdhwcDhwcOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv3DNdhwcDhwcmOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DepthwiseConv3DNdhwcDhwcmOpGenericAdaptorBase::DepthwiseConv3DNdhwcDhwcmOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.depthwise_conv_3d_ndhwc_dhwcm", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DepthwiseConv3DNdhwcDhwcmOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DepthwiseConv3DNdhwcDhwcmOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr DepthwiseConv3DNdhwcDhwcmOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv3DNdhwcDhwcmOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv3DNdhwcDhwcmOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &DepthwiseConv3DNdhwcDhwcmOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange DepthwiseConv3DNdhwcDhwcmOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
DepthwiseConv3DNdhwcDhwcmOpAdaptor::DepthwiseConv3DNdhwcDhwcmOpAdaptor(DepthwiseConv3DNdhwcDhwcmOp op) : DepthwiseConv3DNdhwcDhwcmOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult DepthwiseConv3DNdhwcDhwcmOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.depthwise_conv_3d_ndhwc_dhwcm' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == DepthwiseConv3DNdhwcDhwcmOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == DepthwiseConv3DNdhwcDhwcmOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == DepthwiseConv3DNdhwcDhwcmOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.depthwise_conv_3d_ndhwc_dhwcm' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
    return emitError(loc, "'linalg.depthwise_conv_3d_ndhwc_dhwcm' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
    return emitError(loc, "'linalg.depthwise_conv_3d_ndhwc_dhwcm' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DepthwiseConv3DNdhwcDhwcmOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range DepthwiseConv3DNdhwcDhwcmOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DepthwiseConv3DNdhwcDhwcmOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv3DNdhwcDhwcmOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv3DNdhwcDhwcmOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv3DNdhwcDhwcmOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DepthwiseConv3DNdhwcDhwcmOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DepthwiseConv3DNdhwcDhwcmOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DepthwiseConv3DNdhwcDhwcmOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv3DNdhwcDhwcmOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv3DNdhwcDhwcmOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void DepthwiseConv3DNdhwcDhwcmOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute DepthwiseConv3DNdhwcDhwcmOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute DepthwiseConv3DNdhwcDhwcmOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void DepthwiseConv3DNdhwcDhwcmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, DepthwiseConv3DNdhwcDhwcmOp::getRegionBuilder());
      
}

void DepthwiseConv3DNdhwcDhwcmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, DepthwiseConv3DNdhwcDhwcmOp::getRegionBuilder());
      
}

void DepthwiseConv3DNdhwcDhwcmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void DepthwiseConv3DNdhwcDhwcmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, DepthwiseConv3DNdhwcDhwcmOp::getRegionBuilder());
  
}

::mlir::LogicalResult DepthwiseConv3DNdhwcDhwcmOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps3(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps3(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult DepthwiseConv3DNdhwcDhwcmOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv3DNdhwcDhwcmOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DotOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DotOpGenericAdaptorBase::DotOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.dot", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DotOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DotOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr DotOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &DotOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange DotOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
DotOpAdaptor::DotOpAdaptor(DotOp op) : DotOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult DotOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.dot' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == DotOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.dot' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> DotOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range DotOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DotOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DotOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DotOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DotOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DotOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DotOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DotOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &DotOp::getRegion() {
  return (*this)->getRegion(0);
}

void DotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, DotOp::getRegionBuilder());
      
}

void DotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, DotOp::getRegionBuilder());
      
}

void DotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult DotOp::verifyInvariantsImpl() {
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
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult DotOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DotOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ElemwiseBinaryOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ElemwiseBinaryOpGenericAdaptorBase::ElemwiseBinaryOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.elemwise_binary", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ElemwiseBinaryOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ElemwiseBinaryOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr ElemwiseBinaryOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::linalg::BinaryFnAttr ElemwiseBinaryOpGenericAdaptorBase::getFunAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ElemwiseBinaryOp::getFunAttrName(*odsOpName)).dyn_cast_or_null<::mlir::linalg::BinaryFnAttr>();
  if (!attr)
    attr = ::mlir::linalg::BinaryFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), BinaryFn::add);
  return attr;
}

::mlir::linalg::BinaryFn ElemwiseBinaryOpGenericAdaptorBase::getFun() {
  auto attr = getFunAttr();
    if (!attr)
      return ::mlir::linalg::BinaryFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), BinaryFn::add).getValue();
  return attr.getValue();
}

::mlir::linalg::TypeFnAttr ElemwiseBinaryOpGenericAdaptorBase::getCastAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ElemwiseBinaryOp::getCastAttrName(*odsOpName)).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
  if (!attr)
    attr = ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed);
  return attr;
}

::mlir::linalg::TypeFn ElemwiseBinaryOpGenericAdaptorBase::getCast() {
  auto attr = getCastAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

::mlir::Region &ElemwiseBinaryOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange ElemwiseBinaryOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
ElemwiseBinaryOpAdaptor::ElemwiseBinaryOpAdaptor(ElemwiseBinaryOp op) : ElemwiseBinaryOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ElemwiseBinaryOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_cast;
  ::mlir::Attribute tblgen_fun;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.elemwise_binary' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == ElemwiseBinaryOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == ElemwiseBinaryOp::getCastAttrName(*odsOpName)) {
      tblgen_cast = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == ElemwiseBinaryOp::getFunAttrName(*odsOpName)) {
      tblgen_fun = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.elemwise_binary' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_fun && !((tblgen_fun.isa<::mlir::linalg::BinaryFnAttr>())))
    return emitError(loc, "'linalg.elemwise_binary' op ""attribute 'fun' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6");

  if (tblgen_cast && !((tblgen_cast.isa<::mlir::linalg::TypeFnAttr>())))
    return emitError(loc, "'linalg.elemwise_binary' op ""attribute 'cast' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ElemwiseBinaryOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range ElemwiseBinaryOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ElemwiseBinaryOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ElemwiseBinaryOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ElemwiseBinaryOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ElemwiseBinaryOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> ElemwiseBinaryOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ElemwiseBinaryOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ElemwiseBinaryOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &ElemwiseBinaryOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::linalg::BinaryFnAttr ElemwiseBinaryOp::getFunAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getFunAttrName()).dyn_cast_or_null<::mlir::linalg::BinaryFnAttr>();
}

::mlir::linalg::BinaryFn ElemwiseBinaryOp::getFun() {
  auto attr = getFunAttr();
    if (!attr)
      return ::mlir::linalg::BinaryFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), BinaryFn::add).getValue();
  return attr.getValue();
}

::mlir::linalg::TypeFnAttr ElemwiseBinaryOp::getCastAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCastAttrName()).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
}

::mlir::linalg::TypeFn ElemwiseBinaryOp::getCast() {
  auto attr = getCastAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

void ElemwiseBinaryOp::setFunAttr(::mlir::linalg::BinaryFnAttr attr) {
  (*this)->setAttr(getFunAttrName(), attr);
}

void ElemwiseBinaryOp::setFun(::std::optional<::mlir::linalg::BinaryFn> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getFunAttrName(), ::mlir::linalg::BinaryFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getFunAttrName());
}

void ElemwiseBinaryOp::setCastAttr(::mlir::linalg::TypeFnAttr attr) {
  (*this)->setAttr(getCastAttrName(), attr);
}

void ElemwiseBinaryOp::setCast(::std::optional<::mlir::linalg::TypeFn> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getCastAttrName(), ::mlir::linalg::TypeFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getCastAttrName());
}

::mlir::Attribute ElemwiseBinaryOp::removeFunAttr() {
  return (*this)->removeAttr(getFunAttrName());
}

::mlir::Attribute ElemwiseBinaryOp::removeCastAttr() {
  return (*this)->removeAttr(getCastAttrName());
}

void ElemwiseBinaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, ElemwiseBinaryOp::getRegionBuilder());
      
}

void ElemwiseBinaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, ElemwiseBinaryOp::getRegionBuilder());
      
}

void ElemwiseBinaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void ElemwiseBinaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute fun, Attribute cast, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("fun", fun);
odsState.addAttribute("cast", cast);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, ElemwiseBinaryOp::getRegionBuilder());
  
}

::mlir::LogicalResult ElemwiseBinaryOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_cast;
  ::mlir::Attribute tblgen_fun;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getCastAttrName()) {
      tblgen_cast = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getFunAttrName()) {
      tblgen_fun = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps5(*this, tblgen_fun, "fun")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps4(*this, tblgen_cast, "cast")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult ElemwiseBinaryOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::ElemwiseBinaryOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ElemwiseUnaryOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ElemwiseUnaryOpGenericAdaptorBase::ElemwiseUnaryOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.elemwise_unary", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ElemwiseUnaryOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ElemwiseUnaryOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr ElemwiseUnaryOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::linalg::UnaryFnAttr ElemwiseUnaryOpGenericAdaptorBase::getFunAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ElemwiseUnaryOp::getFunAttrName(*odsOpName)).dyn_cast_or_null<::mlir::linalg::UnaryFnAttr>();
  if (!attr)
    attr = ::mlir::linalg::UnaryFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), UnaryFn::exp);
  return attr;
}

::mlir::linalg::UnaryFn ElemwiseUnaryOpGenericAdaptorBase::getFun() {
  auto attr = getFunAttr();
    if (!attr)
      return ::mlir::linalg::UnaryFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), UnaryFn::exp).getValue();
  return attr.getValue();
}

::mlir::linalg::TypeFnAttr ElemwiseUnaryOpGenericAdaptorBase::getCastAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ElemwiseUnaryOp::getCastAttrName(*odsOpName)).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
  if (!attr)
    attr = ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed);
  return attr;
}

::mlir::linalg::TypeFn ElemwiseUnaryOpGenericAdaptorBase::getCast() {
  auto attr = getCastAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

::mlir::Region &ElemwiseUnaryOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange ElemwiseUnaryOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
ElemwiseUnaryOpAdaptor::ElemwiseUnaryOpAdaptor(ElemwiseUnaryOp op) : ElemwiseUnaryOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ElemwiseUnaryOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_cast;
  ::mlir::Attribute tblgen_fun;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.elemwise_unary' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == ElemwiseUnaryOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == ElemwiseUnaryOp::getCastAttrName(*odsOpName)) {
      tblgen_cast = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == ElemwiseUnaryOp::getFunAttrName(*odsOpName)) {
      tblgen_fun = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.elemwise_unary' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_fun && !((tblgen_fun.isa<::mlir::linalg::UnaryFnAttr>())))
    return emitError(loc, "'linalg.elemwise_unary' op ""attribute 'fun' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5");

  if (tblgen_cast && !((tblgen_cast.isa<::mlir::linalg::TypeFnAttr>())))
    return emitError(loc, "'linalg.elemwise_unary' op ""attribute 'cast' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ElemwiseUnaryOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range ElemwiseUnaryOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ElemwiseUnaryOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ElemwiseUnaryOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ElemwiseUnaryOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ElemwiseUnaryOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> ElemwiseUnaryOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ElemwiseUnaryOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ElemwiseUnaryOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &ElemwiseUnaryOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::linalg::UnaryFnAttr ElemwiseUnaryOp::getFunAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getFunAttrName()).dyn_cast_or_null<::mlir::linalg::UnaryFnAttr>();
}

::mlir::linalg::UnaryFn ElemwiseUnaryOp::getFun() {
  auto attr = getFunAttr();
    if (!attr)
      return ::mlir::linalg::UnaryFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), UnaryFn::exp).getValue();
  return attr.getValue();
}

::mlir::linalg::TypeFnAttr ElemwiseUnaryOp::getCastAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCastAttrName()).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
}

::mlir::linalg::TypeFn ElemwiseUnaryOp::getCast() {
  auto attr = getCastAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

void ElemwiseUnaryOp::setFunAttr(::mlir::linalg::UnaryFnAttr attr) {
  (*this)->setAttr(getFunAttrName(), attr);
}

void ElemwiseUnaryOp::setFun(::std::optional<::mlir::linalg::UnaryFn> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getFunAttrName(), ::mlir::linalg::UnaryFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getFunAttrName());
}

void ElemwiseUnaryOp::setCastAttr(::mlir::linalg::TypeFnAttr attr) {
  (*this)->setAttr(getCastAttrName(), attr);
}

void ElemwiseUnaryOp::setCast(::std::optional<::mlir::linalg::TypeFn> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getCastAttrName(), ::mlir::linalg::TypeFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getCastAttrName());
}

::mlir::Attribute ElemwiseUnaryOp::removeFunAttr() {
  return (*this)->removeAttr(getFunAttrName());
}

::mlir::Attribute ElemwiseUnaryOp::removeCastAttr() {
  return (*this)->removeAttr(getCastAttrName());
}

void ElemwiseUnaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, ElemwiseUnaryOp::getRegionBuilder());
      
}

void ElemwiseUnaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, ElemwiseUnaryOp::getRegionBuilder());
      
}

void ElemwiseUnaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void ElemwiseUnaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute fun, Attribute cast, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("fun", fun);
odsState.addAttribute("cast", cast);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, ElemwiseUnaryOp::getRegionBuilder());
  
}

::mlir::LogicalResult ElemwiseUnaryOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_cast;
  ::mlir::Attribute tblgen_fun;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getCastAttrName()) {
      tblgen_cast = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getFunAttrName()) {
      tblgen_fun = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps6(*this, tblgen_fun, "fun")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps4(*this, tblgen_cast, "cast")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult ElemwiseUnaryOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::ElemwiseUnaryOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::FillOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FillOpGenericAdaptorBase::FillOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.fill", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FillOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FillOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr FillOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &FillOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange FillOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
FillOpAdaptor::FillOpAdaptor(FillOp op) : FillOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult FillOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.fill' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == FillOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.fill' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> FillOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range FillOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range FillOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range FillOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange FillOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange FillOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> FillOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range FillOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range FillOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &FillOp::getRegion() {
  return (*this)->getRegion(0);
}

void FillOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, FillOp::getRegionBuilder());
      
}

void FillOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, FillOp::getRegionBuilder());
      
}

void FillOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult FillOp::verifyInvariantsImpl() {
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
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult FillOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::FillOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::FillRng2DOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FillRng2DOpGenericAdaptorBase::FillRng2DOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.fill_rng_2d", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FillRng2DOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FillRng2DOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr FillRng2DOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &FillRng2DOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange FillRng2DOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
FillRng2DOpAdaptor::FillRng2DOpAdaptor(FillRng2DOp op) : FillRng2DOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult FillRng2DOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.fill_rng_2d' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == FillRng2DOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.fill_rng_2d' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> FillRng2DOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range FillRng2DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range FillRng2DOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range FillRng2DOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange FillRng2DOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange FillRng2DOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> FillRng2DOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range FillRng2DOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range FillRng2DOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &FillRng2DOp::getRegion() {
  return (*this)->getRegion(0);
}

void FillRng2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, FillRng2DOp::getRegionBuilder());
      
}

void FillRng2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, FillRng2DOp::getRegionBuilder());
      
}

void FillRng2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult FillRng2DOp::verifyInvariantsImpl() {
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
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult FillRng2DOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::FillRng2DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::GenericOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GenericOpGenericAdaptorBase::GenericOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.generic", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GenericOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, GenericOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr GenericOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr GenericOpGenericAdaptorBase::getIndexingMapsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, GenericOp::getIndexingMapsAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr GenericOpGenericAdaptorBase::getIndexingMaps() {
  auto attr = getIndexingMapsAttr();
  return attr;
}

::mlir::ArrayAttr GenericOpGenericAdaptorBase::getIteratorTypesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, GenericOp::getIteratorTypesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr GenericOpGenericAdaptorBase::getIteratorTypes() {
  auto attr = getIteratorTypesAttr();
  return attr;
}

::mlir::StringAttr GenericOpGenericAdaptorBase::getDocAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, GenericOp::getDocAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > GenericOpGenericAdaptorBase::getDoc() {
  auto attr = getDocAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::StringAttr GenericOpGenericAdaptorBase::getLibraryCallAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, GenericOp::getLibraryCallAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > GenericOpGenericAdaptorBase::getLibraryCall() {
  auto attr = getLibraryCallAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::Region &GenericOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange GenericOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
GenericOpAdaptor::GenericOpAdaptor(GenericOp op) : GenericOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult GenericOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_indexing_maps;
  ::mlir::Attribute tblgen_doc;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.generic' op ""requires attribute 'indexing_maps'");
    if (namedAttrIt->getName() == GenericOp::getIndexingMapsAttrName(*odsOpName)) {
      tblgen_indexing_maps = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == GenericOp::getDocAttrName(*odsOpName)) {
      tblgen_doc = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_iterator_types;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.generic' op ""requires attribute 'iterator_types'");
    if (namedAttrIt->getName() == GenericOp::getIteratorTypesAttrName(*odsOpName)) {
      tblgen_iterator_types = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_library_call;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.generic' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == GenericOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == GenericOp::getLibraryCallAttrName(*odsOpName)) {
      tblgen_library_call = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.generic' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_indexing_maps && !(((tblgen_indexing_maps.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_indexing_maps.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::AffineMapAttr>())); }))))
    return emitError(loc, "'linalg.generic' op ""attribute 'indexing_maps' failed to satisfy constraint: AffineMap array attribute");

  if (tblgen_iterator_types && !(((tblgen_iterator_types.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_iterator_types.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::linalg::IteratorTypeAttr>())); }))))
    return emitError(loc, "'linalg.generic' op ""attribute 'iterator_types' failed to satisfy constraint: Iterator type should be an enum.");

  if (tblgen_doc && !((tblgen_doc.isa<::mlir::StringAttr>())))
    return emitError(loc, "'linalg.generic' op ""attribute 'doc' failed to satisfy constraint: string attribute");

  if (tblgen_library_call && !((tblgen_library_call.isa<::mlir::StringAttr>())))
    return emitError(loc, "'linalg.generic' op ""attribute 'library_call' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GenericOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range GenericOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range GenericOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range GenericOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange GenericOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange GenericOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> GenericOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range GenericOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range GenericOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &GenericOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::ArrayAttr GenericOp::getIndexingMapsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getIndexingMapsAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GenericOp::getIndexingMaps() {
  auto attr = getIndexingMapsAttr();
  return attr;
}

::mlir::ArrayAttr GenericOp::getIteratorTypesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getIteratorTypesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GenericOp::getIteratorTypes() {
  auto attr = getIteratorTypesAttr();
  return attr;
}

::mlir::StringAttr GenericOp::getDocAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getDocAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > GenericOp::getDoc() {
  auto attr = getDocAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::StringAttr GenericOp::getLibraryCallAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getLibraryCallAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > GenericOp::getLibraryCall() {
  auto attr = getLibraryCallAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

void GenericOp::setIndexingMapsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getIndexingMapsAttrName(), attr);
}

void GenericOp::setIteratorTypesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getIteratorTypesAttrName(), attr);
}

void GenericOp::setDocAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getDocAttrName(), attr);
}

void GenericOp::setDoc(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getDocAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getDocAttrName());
}

void GenericOp::setLibraryCallAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getLibraryCallAttrName(), attr);
}

void GenericOp::setLibraryCall(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getLibraryCallAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getLibraryCallAttrName());
}

::mlir::Attribute GenericOp::removeDocAttr() {
  return (*this)->removeAttr(getDocAttrName());
}

::mlir::Attribute GenericOp::removeLibraryCallAttr() {
  return (*this)->removeAttr(getLibraryCallAttrName());
}

void GenericOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result_tensors, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::ArrayAttr indexing_maps, ::mlir::ArrayAttr iterator_types, /*optional*/::mlir::StringAttr doc, /*optional*/::mlir::StringAttr library_call) {
  odsState.addOperands(inputs);
  odsState.addOperands(outputs);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(outputs.size())}));
  odsState.addAttribute(getIndexingMapsAttrName(odsState.name), indexing_maps);
  odsState.addAttribute(getIteratorTypesAttrName(odsState.name), iterator_types);
  if (doc) {
    odsState.addAttribute(getDocAttrName(odsState.name), doc);
  }
  if (library_call) {
    odsState.addAttribute(getLibraryCallAttrName(odsState.name), library_call);
  }
  (void)odsState.addRegion();
  odsState.addTypes(result_tensors);
}

void GenericOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GenericOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_indexing_maps;
  ::mlir::Attribute tblgen_doc;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'indexing_maps'");
    if (namedAttrIt->getName() == getIndexingMapsAttrName()) {
      tblgen_indexing_maps = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDocAttrName()) {
      tblgen_doc = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_iterator_types;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'iterator_types'");
    if (namedAttrIt->getName() == getIteratorTypesAttrName()) {
      tblgen_iterator_types = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_library_call;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getLibraryCallAttrName()) {
      tblgen_library_call = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps7(*this, tblgen_indexing_maps, "indexing_maps")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps8(*this, tblgen_iterator_types, "iterator_types")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps9(*this, tblgen_doc, "doc")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps9(*this, tblgen_library_call, "library_call")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult GenericOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::GenericOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MapOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
MapOpGenericAdaptorBase::MapOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.map", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> MapOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true, false};
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

::mlir::DictionaryAttr MapOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &MapOpGenericAdaptorBase::getMapper() {
  return *odsRegions[0];
}

::mlir::RegionRange MapOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
MapOpAdaptor::MapOpAdaptor(MapOp op) : MapOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult MapOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MapOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
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

::mlir::Operation::operand_range MapOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MapOp::getInputs() {
  return getODSOperands(0);
}

::mlir::TypedValue<::mlir::ShapedType> MapOp::getInit() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::ShapedType>>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange MapOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MapOp::getInitMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MapOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range MapOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range MapOp::getResult() {
  return getODSResults(0);
}

::mlir::Region &MapOp::getMapper() {
  return (*this)->getRegion(0);
}

void MapOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result, ::mlir::ValueRange inputs, ::mlir::Value init) {
  odsState.addOperands(inputs);
  odsState.addOperands(init);
  (void)odsState.addRegion();
  odsState.addTypes(result);
}

void MapOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MapOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps1(*this, region, "mapper", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult MapOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::MapOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatmulOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
MatmulOpGenericAdaptorBase::MatmulOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.matmul", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> MatmulOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MatmulOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr MatmulOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::linalg::TypeFnAttr MatmulOpGenericAdaptorBase::getCastAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, MatmulOp::getCastAttrName(*odsOpName)).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
  if (!attr)
    attr = ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed);
  return attr;
}

::mlir::linalg::TypeFn MatmulOpGenericAdaptorBase::getCast() {
  auto attr = getCastAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

::mlir::Region &MatmulOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange MatmulOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
MatmulOpAdaptor::MatmulOpAdaptor(MatmulOp op) : MatmulOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult MatmulOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_cast;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.matmul' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == MatmulOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == MatmulOp::getCastAttrName(*odsOpName)) {
      tblgen_cast = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.matmul' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_cast && !((tblgen_cast.isa<::mlir::linalg::TypeFnAttr>())))
    return emitError(loc, "'linalg.matmul' op ""attribute 'cast' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MatmulOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range MatmulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatmulOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatmulOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatmulOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MatmulOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> MatmulOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range MatmulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range MatmulOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &MatmulOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::linalg::TypeFnAttr MatmulOp::getCastAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCastAttrName()).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
}

::mlir::linalg::TypeFn MatmulOp::getCast() {
  auto attr = getCastAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

void MatmulOp::setCastAttr(::mlir::linalg::TypeFnAttr attr) {
  (*this)->setAttr(getCastAttrName(), attr);
}

void MatmulOp::setCast(::std::optional<::mlir::linalg::TypeFn> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getCastAttrName(), ::mlir::linalg::TypeFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getCastAttrName());
}

::mlir::Attribute MatmulOp::removeCastAttr() {
  return (*this)->removeAttr(getCastAttrName());
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, MatmulOp::getRegionBuilder());
      
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, MatmulOp::getRegionBuilder());
      
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute cast, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("cast", cast);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, MatmulOp::getRegionBuilder());
  
}

::mlir::LogicalResult MatmulOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_cast;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getCastAttrName()) {
      tblgen_cast = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps4(*this, tblgen_cast, "cast")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult MatmulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::MatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatmulTransposeBOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
MatmulTransposeBOpGenericAdaptorBase::MatmulTransposeBOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.matmul_transpose_b", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> MatmulTransposeBOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MatmulTransposeBOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr MatmulTransposeBOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::linalg::TypeFnAttr MatmulTransposeBOpGenericAdaptorBase::getCastAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, MatmulTransposeBOp::getCastAttrName(*odsOpName)).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
  if (!attr)
    attr = ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed);
  return attr;
}

::mlir::linalg::TypeFn MatmulTransposeBOpGenericAdaptorBase::getCast() {
  auto attr = getCastAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

::mlir::Region &MatmulTransposeBOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange MatmulTransposeBOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
MatmulTransposeBOpAdaptor::MatmulTransposeBOpAdaptor(MatmulTransposeBOp op) : MatmulTransposeBOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult MatmulTransposeBOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_cast;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.matmul_transpose_b' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == MatmulTransposeBOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == MatmulTransposeBOp::getCastAttrName(*odsOpName)) {
      tblgen_cast = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.matmul_transpose_b' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_cast && !((tblgen_cast.isa<::mlir::linalg::TypeFnAttr>())))
    return emitError(loc, "'linalg.matmul_transpose_b' op ""attribute 'cast' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MatmulTransposeBOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range MatmulTransposeBOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatmulTransposeBOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatmulTransposeBOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatmulTransposeBOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MatmulTransposeBOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> MatmulTransposeBOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range MatmulTransposeBOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range MatmulTransposeBOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &MatmulTransposeBOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::linalg::TypeFnAttr MatmulTransposeBOp::getCastAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCastAttrName()).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
}

::mlir::linalg::TypeFn MatmulTransposeBOp::getCast() {
  auto attr = getCastAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

void MatmulTransposeBOp::setCastAttr(::mlir::linalg::TypeFnAttr attr) {
  (*this)->setAttr(getCastAttrName(), attr);
}

void MatmulTransposeBOp::setCast(::std::optional<::mlir::linalg::TypeFn> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getCastAttrName(), ::mlir::linalg::TypeFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue));
    (*this)->removeAttr(getCastAttrName());
}

::mlir::Attribute MatmulTransposeBOp::removeCastAttr() {
  return (*this)->removeAttr(getCastAttrName());
}

void MatmulTransposeBOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, MatmulTransposeBOp::getRegionBuilder());
      
}

void MatmulTransposeBOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, MatmulTransposeBOp::getRegionBuilder());
      
}

void MatmulTransposeBOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void MatmulTransposeBOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute cast, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("cast", cast);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, MatmulTransposeBOp::getRegionBuilder());
  
}

::mlir::LogicalResult MatmulTransposeBOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_cast;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getCastAttrName()) {
      tblgen_cast = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps4(*this, tblgen_cast, "cast")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult MatmulTransposeBOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::MatmulTransposeBOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatmulUnsignedOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
MatmulUnsignedOpGenericAdaptorBase::MatmulUnsignedOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.matmul_unsigned", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> MatmulUnsignedOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MatmulUnsignedOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr MatmulUnsignedOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &MatmulUnsignedOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange MatmulUnsignedOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
MatmulUnsignedOpAdaptor::MatmulUnsignedOpAdaptor(MatmulUnsignedOp op) : MatmulUnsignedOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult MatmulUnsignedOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.matmul_unsigned' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == MatmulUnsignedOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.matmul_unsigned' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> MatmulUnsignedOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range MatmulUnsignedOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatmulUnsignedOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatmulUnsignedOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatmulUnsignedOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MatmulUnsignedOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> MatmulUnsignedOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range MatmulUnsignedOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range MatmulUnsignedOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &MatmulUnsignedOp::getRegion() {
  return (*this)->getRegion(0);
}

void MatmulUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, MatmulUnsignedOp::getRegionBuilder());
      
}

void MatmulUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, MatmulUnsignedOp::getRegionBuilder());
      
}

void MatmulUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult MatmulUnsignedOp::verifyInvariantsImpl() {
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
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult MatmulUnsignedOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::MatmulUnsignedOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatvecOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
MatvecOpGenericAdaptorBase::MatvecOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.matvec", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> MatvecOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MatvecOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr MatvecOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &MatvecOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange MatvecOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
MatvecOpAdaptor::MatvecOpAdaptor(MatvecOp op) : MatvecOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult MatvecOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.matvec' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == MatvecOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.matvec' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> MatvecOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range MatvecOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatvecOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatvecOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatvecOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MatvecOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> MatvecOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range MatvecOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range MatvecOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &MatvecOp::getRegion() {
  return (*this)->getRegion(0);
}

void MatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, MatvecOp::getRegionBuilder());
      
}

void MatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, MatvecOp::getRegionBuilder());
      
}

void MatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult MatvecOp::verifyInvariantsImpl() {
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
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult MatvecOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::MatvecOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Mmt4DOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
Mmt4DOpGenericAdaptorBase::Mmt4DOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.mmt4d", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> Mmt4DOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Mmt4DOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr Mmt4DOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &Mmt4DOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange Mmt4DOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
Mmt4DOpAdaptor::Mmt4DOpAdaptor(Mmt4DOp op) : Mmt4DOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult Mmt4DOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.mmt4d' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == Mmt4DOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.mmt4d' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> Mmt4DOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range Mmt4DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Mmt4DOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Mmt4DOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Mmt4DOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Mmt4DOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Mmt4DOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Mmt4DOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Mmt4DOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &Mmt4DOp::getRegion() {
  return (*this)->getRegion(0);
}

void Mmt4DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, Mmt4DOp::getRegionBuilder());
      
}

void Mmt4DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, Mmt4DOp::getRegionBuilder());
      
}

void Mmt4DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult Mmt4DOp::verifyInvariantsImpl() {
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
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult Mmt4DOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Mmt4DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNchwMaxOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PoolingNchwMaxOpGenericAdaptorBase::PoolingNchwMaxOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_nchw_max", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PoolingNchwMaxOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNchwMaxOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr PoolingNchwMaxOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNchwMaxOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNchwMaxOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &PoolingNchwMaxOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange PoolingNchwMaxOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
PoolingNchwMaxOpAdaptor::PoolingNchwMaxOpAdaptor(PoolingNchwMaxOp op) : PoolingNchwMaxOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PoolingNchwMaxOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.pooling_nchw_max' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == PoolingNchwMaxOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == PoolingNchwMaxOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PoolingNchwMaxOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_nchw_max' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.pooling_nchw_max' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.pooling_nchw_max' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNchwMaxOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range PoolingNchwMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNchwMaxOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNchwMaxOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNchwMaxOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNchwMaxOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNchwMaxOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNchwMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNchwMaxOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNchwMaxOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNchwMaxOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNchwMaxOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute PoolingNchwMaxOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNchwMaxOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNchwMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, PoolingNchwMaxOp::getRegionBuilder());
      
}

void PoolingNchwMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, PoolingNchwMaxOp::getRegionBuilder());
      
}

void PoolingNchwMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNchwMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, PoolingNchwMaxOp::getRegionBuilder());
  
}

::mlir::LogicalResult PoolingNchwMaxOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult PoolingNchwMaxOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNchwMaxOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNchwSumOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PoolingNchwSumOpGenericAdaptorBase::PoolingNchwSumOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_nchw_sum", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PoolingNchwSumOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNchwSumOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr PoolingNchwSumOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNchwSumOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNchwSumOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwSumOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwSumOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNchwSumOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwSumOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &PoolingNchwSumOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange PoolingNchwSumOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
PoolingNchwSumOpAdaptor::PoolingNchwSumOpAdaptor(PoolingNchwSumOp op) : PoolingNchwSumOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PoolingNchwSumOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.pooling_nchw_sum' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == PoolingNchwSumOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == PoolingNchwSumOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PoolingNchwSumOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_nchw_sum' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.pooling_nchw_sum' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.pooling_nchw_sum' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNchwSumOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range PoolingNchwSumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNchwSumOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNchwSumOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNchwSumOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNchwSumOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNchwSumOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNchwSumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNchwSumOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNchwSumOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNchwSumOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNchwSumOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwSumOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNchwSumOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNchwSumOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNchwSumOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute PoolingNchwSumOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNchwSumOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNchwSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, PoolingNchwSumOp::getRegionBuilder());
      
}

void PoolingNchwSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, PoolingNchwSumOp::getRegionBuilder());
      
}

void PoolingNchwSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNchwSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, PoolingNchwSumOp::getRegionBuilder());
  
}

::mlir::LogicalResult PoolingNchwSumOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult PoolingNchwSumOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNchwSumOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNcwMaxOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PoolingNcwMaxOpGenericAdaptorBase::PoolingNcwMaxOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_ncw_max", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PoolingNcwMaxOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNcwMaxOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr PoolingNcwMaxOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNcwMaxOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNcwMaxOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNcwMaxOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNcwMaxOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNcwMaxOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNcwMaxOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &PoolingNcwMaxOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange PoolingNcwMaxOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
PoolingNcwMaxOpAdaptor::PoolingNcwMaxOpAdaptor(PoolingNcwMaxOp op) : PoolingNcwMaxOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PoolingNcwMaxOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.pooling_ncw_max' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == PoolingNcwMaxOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == PoolingNcwMaxOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PoolingNcwMaxOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_ncw_max' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.pooling_ncw_max' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.pooling_ncw_max' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNcwMaxOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range PoolingNcwMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNcwMaxOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNcwMaxOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNcwMaxOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNcwMaxOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNcwMaxOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNcwMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNcwMaxOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNcwMaxOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNcwMaxOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNcwMaxOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNcwMaxOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNcwMaxOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNcwMaxOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNcwMaxOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute PoolingNcwMaxOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNcwMaxOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNcwMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, PoolingNcwMaxOp::getRegionBuilder());
      
}

void PoolingNcwMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, PoolingNcwMaxOp::getRegionBuilder());
      
}

void PoolingNcwMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNcwMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, PoolingNcwMaxOp::getRegionBuilder());
  
}

::mlir::LogicalResult PoolingNcwMaxOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult PoolingNcwMaxOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNcwMaxOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNcwSumOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PoolingNcwSumOpGenericAdaptorBase::PoolingNcwSumOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_ncw_sum", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PoolingNcwSumOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNcwSumOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr PoolingNcwSumOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNcwSumOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNcwSumOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNcwSumOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNcwSumOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNcwSumOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNcwSumOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &PoolingNcwSumOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange PoolingNcwSumOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
PoolingNcwSumOpAdaptor::PoolingNcwSumOpAdaptor(PoolingNcwSumOp op) : PoolingNcwSumOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PoolingNcwSumOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.pooling_ncw_sum' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == PoolingNcwSumOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == PoolingNcwSumOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PoolingNcwSumOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_ncw_sum' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.pooling_ncw_sum' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.pooling_ncw_sum' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNcwSumOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range PoolingNcwSumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNcwSumOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNcwSumOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNcwSumOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNcwSumOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNcwSumOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNcwSumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNcwSumOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNcwSumOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNcwSumOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNcwSumOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNcwSumOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNcwSumOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNcwSumOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNcwSumOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute PoolingNcwSumOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNcwSumOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNcwSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, PoolingNcwSumOp::getRegionBuilder());
      
}

void PoolingNcwSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, PoolingNcwSumOp::getRegionBuilder());
      
}

void PoolingNcwSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNcwSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, PoolingNcwSumOp::getRegionBuilder());
  
}

::mlir::LogicalResult PoolingNcwSumOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult PoolingNcwSumOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNcwSumOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNdhwcMaxOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PoolingNdhwcMaxOpGenericAdaptorBase::PoolingNdhwcMaxOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_ndhwc_max", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PoolingNdhwcMaxOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNdhwcMaxOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr PoolingNdhwcMaxOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNdhwcMaxOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNdhwcMaxOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &PoolingNdhwcMaxOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange PoolingNdhwcMaxOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
PoolingNdhwcMaxOpAdaptor::PoolingNdhwcMaxOpAdaptor(PoolingNdhwcMaxOp op) : PoolingNdhwcMaxOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PoolingNdhwcMaxOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.pooling_ndhwc_max' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == PoolingNdhwcMaxOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == PoolingNdhwcMaxOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PoolingNdhwcMaxOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_ndhwc_max' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
    return emitError(loc, "'linalg.pooling_ndhwc_max' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
    return emitError(loc, "'linalg.pooling_ndhwc_max' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNdhwcMaxOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range PoolingNdhwcMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNdhwcMaxOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNdhwcMaxOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNdhwcMaxOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNdhwcMaxOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNdhwcMaxOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNdhwcMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNdhwcMaxOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNdhwcMaxOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNdhwcMaxOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNdhwcMaxOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute PoolingNdhwcMaxOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNdhwcMaxOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNdhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, PoolingNdhwcMaxOp::getRegionBuilder());
      
}

void PoolingNdhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, PoolingNdhwcMaxOp::getRegionBuilder());
      
}

void PoolingNdhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNdhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, PoolingNdhwcMaxOp::getRegionBuilder());
  
}

::mlir::LogicalResult PoolingNdhwcMaxOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps3(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps3(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult PoolingNdhwcMaxOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNdhwcMaxOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNdhwcMinOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PoolingNdhwcMinOpGenericAdaptorBase::PoolingNdhwcMinOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_ndhwc_min", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PoolingNdhwcMinOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNdhwcMinOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr PoolingNdhwcMinOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNdhwcMinOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNdhwcMinOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &PoolingNdhwcMinOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange PoolingNdhwcMinOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
PoolingNdhwcMinOpAdaptor::PoolingNdhwcMinOpAdaptor(PoolingNdhwcMinOp op) : PoolingNdhwcMinOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PoolingNdhwcMinOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.pooling_ndhwc_min' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == PoolingNdhwcMinOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == PoolingNdhwcMinOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PoolingNdhwcMinOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_ndhwc_min' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
    return emitError(loc, "'linalg.pooling_ndhwc_min' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
    return emitError(loc, "'linalg.pooling_ndhwc_min' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNdhwcMinOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range PoolingNdhwcMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNdhwcMinOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNdhwcMinOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNdhwcMinOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNdhwcMinOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNdhwcMinOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNdhwcMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNdhwcMinOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNdhwcMinOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNdhwcMinOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNdhwcMinOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute PoolingNdhwcMinOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNdhwcMinOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNdhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, PoolingNdhwcMinOp::getRegionBuilder());
      
}

void PoolingNdhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, PoolingNdhwcMinOp::getRegionBuilder());
      
}

void PoolingNdhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNdhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, PoolingNdhwcMinOp::getRegionBuilder());
  
}

::mlir::LogicalResult PoolingNdhwcMinOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps3(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps3(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult PoolingNdhwcMinOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNdhwcMinOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNdhwcSumOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PoolingNdhwcSumOpGenericAdaptorBase::PoolingNdhwcSumOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_ndhwc_sum", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PoolingNdhwcSumOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNdhwcSumOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr PoolingNdhwcSumOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNdhwcSumOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNdhwcSumOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &PoolingNdhwcSumOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange PoolingNdhwcSumOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
PoolingNdhwcSumOpAdaptor::PoolingNdhwcSumOpAdaptor(PoolingNdhwcSumOp op) : PoolingNdhwcSumOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PoolingNdhwcSumOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.pooling_ndhwc_sum' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == PoolingNdhwcSumOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == PoolingNdhwcSumOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PoolingNdhwcSumOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_ndhwc_sum' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
    return emitError(loc, "'linalg.pooling_ndhwc_sum' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
    return emitError(loc, "'linalg.pooling_ndhwc_sum' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNdhwcSumOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range PoolingNdhwcSumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNdhwcSumOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNdhwcSumOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNdhwcSumOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNdhwcSumOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNdhwcSumOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNdhwcSumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNdhwcSumOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNdhwcSumOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNdhwcSumOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNdhwcSumOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute PoolingNdhwcSumOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNdhwcSumOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNdhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, PoolingNdhwcSumOp::getRegionBuilder());
      
}

void PoolingNdhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, PoolingNdhwcSumOp::getRegionBuilder());
      
}

void PoolingNdhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNdhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, PoolingNdhwcSumOp::getRegionBuilder());
  
}

::mlir::LogicalResult PoolingNdhwcSumOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps3(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps3(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult PoolingNdhwcSumOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNdhwcSumOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcMaxOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PoolingNhwcMaxOpGenericAdaptorBase::PoolingNhwcMaxOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_nhwc_max", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PoolingNhwcMaxOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNhwcMaxOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr PoolingNhwcMaxOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNhwcMaxOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNhwcMaxOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &PoolingNhwcMaxOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange PoolingNhwcMaxOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
PoolingNhwcMaxOpAdaptor::PoolingNhwcMaxOpAdaptor(PoolingNhwcMaxOp op) : PoolingNhwcMaxOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PoolingNhwcMaxOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.pooling_nhwc_max' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == PoolingNhwcMaxOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == PoolingNhwcMaxOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PoolingNhwcMaxOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_nhwc_max' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.pooling_nhwc_max' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.pooling_nhwc_max' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNhwcMaxOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range PoolingNhwcMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNhwcMaxOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNhwcMaxOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNhwcMaxOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMaxOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNhwcMaxOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNhwcMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNhwcMaxOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNhwcMaxOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNhwcMaxOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNhwcMaxOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute PoolingNhwcMaxOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNhwcMaxOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, PoolingNhwcMaxOp::getRegionBuilder());
      
}

void PoolingNhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, PoolingNhwcMaxOp::getRegionBuilder());
      
}

void PoolingNhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, PoolingNhwcMaxOp::getRegionBuilder());
  
}

::mlir::LogicalResult PoolingNhwcMaxOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult PoolingNhwcMaxOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcMaxOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcMaxUnsignedOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PoolingNhwcMaxUnsignedOpGenericAdaptorBase::PoolingNhwcMaxUnsignedOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_nhwc_max_unsigned", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PoolingNhwcMaxUnsignedOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNhwcMaxUnsignedOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr PoolingNhwcMaxUnsignedOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNhwcMaxUnsignedOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNhwcMaxUnsignedOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &PoolingNhwcMaxUnsignedOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange PoolingNhwcMaxUnsignedOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
PoolingNhwcMaxUnsignedOpAdaptor::PoolingNhwcMaxUnsignedOpAdaptor(PoolingNhwcMaxUnsignedOp op) : PoolingNhwcMaxUnsignedOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PoolingNhwcMaxUnsignedOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.pooling_nhwc_max_unsigned' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == PoolingNhwcMaxUnsignedOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == PoolingNhwcMaxUnsignedOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PoolingNhwcMaxUnsignedOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_nhwc_max_unsigned' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.pooling_nhwc_max_unsigned' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.pooling_nhwc_max_unsigned' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNhwcMaxUnsignedOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range PoolingNhwcMaxUnsignedOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNhwcMaxUnsignedOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNhwcMaxUnsignedOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNhwcMaxUnsignedOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMaxUnsignedOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNhwcMaxUnsignedOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNhwcMaxUnsignedOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNhwcMaxUnsignedOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNhwcMaxUnsignedOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNhwcMaxUnsignedOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNhwcMaxUnsignedOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute PoolingNhwcMaxUnsignedOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNhwcMaxUnsignedOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNhwcMaxUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, PoolingNhwcMaxUnsignedOp::getRegionBuilder());
      
}

void PoolingNhwcMaxUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, PoolingNhwcMaxUnsignedOp::getRegionBuilder());
      
}

void PoolingNhwcMaxUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNhwcMaxUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, PoolingNhwcMaxUnsignedOp::getRegionBuilder());
  
}

::mlir::LogicalResult PoolingNhwcMaxUnsignedOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult PoolingNhwcMaxUnsignedOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcMaxUnsignedOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcMinOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PoolingNhwcMinOpGenericAdaptorBase::PoolingNhwcMinOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_nhwc_min", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PoolingNhwcMinOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNhwcMinOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr PoolingNhwcMinOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNhwcMinOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNhwcMinOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &PoolingNhwcMinOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange PoolingNhwcMinOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
PoolingNhwcMinOpAdaptor::PoolingNhwcMinOpAdaptor(PoolingNhwcMinOp op) : PoolingNhwcMinOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PoolingNhwcMinOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.pooling_nhwc_min' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == PoolingNhwcMinOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == PoolingNhwcMinOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PoolingNhwcMinOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_nhwc_min' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.pooling_nhwc_min' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.pooling_nhwc_min' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNhwcMinOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range PoolingNhwcMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNhwcMinOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNhwcMinOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNhwcMinOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMinOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNhwcMinOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNhwcMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNhwcMinOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNhwcMinOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNhwcMinOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNhwcMinOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute PoolingNhwcMinOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNhwcMinOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, PoolingNhwcMinOp::getRegionBuilder());
      
}

void PoolingNhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, PoolingNhwcMinOp::getRegionBuilder());
      
}

void PoolingNhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, PoolingNhwcMinOp::getRegionBuilder());
  
}

::mlir::LogicalResult PoolingNhwcMinOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult PoolingNhwcMinOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcMinOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcMinUnsignedOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PoolingNhwcMinUnsignedOpGenericAdaptorBase::PoolingNhwcMinUnsignedOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_nhwc_min_unsigned", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PoolingNhwcMinUnsignedOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNhwcMinUnsignedOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr PoolingNhwcMinUnsignedOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNhwcMinUnsignedOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNhwcMinUnsignedOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &PoolingNhwcMinUnsignedOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange PoolingNhwcMinUnsignedOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
PoolingNhwcMinUnsignedOpAdaptor::PoolingNhwcMinUnsignedOpAdaptor(PoolingNhwcMinUnsignedOp op) : PoolingNhwcMinUnsignedOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PoolingNhwcMinUnsignedOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.pooling_nhwc_min_unsigned' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == PoolingNhwcMinUnsignedOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == PoolingNhwcMinUnsignedOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PoolingNhwcMinUnsignedOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_nhwc_min_unsigned' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.pooling_nhwc_min_unsigned' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.pooling_nhwc_min_unsigned' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNhwcMinUnsignedOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range PoolingNhwcMinUnsignedOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNhwcMinUnsignedOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNhwcMinUnsignedOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNhwcMinUnsignedOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMinUnsignedOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNhwcMinUnsignedOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNhwcMinUnsignedOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNhwcMinUnsignedOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNhwcMinUnsignedOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNhwcMinUnsignedOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNhwcMinUnsignedOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute PoolingNhwcMinUnsignedOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNhwcMinUnsignedOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNhwcMinUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, PoolingNhwcMinUnsignedOp::getRegionBuilder());
      
}

void PoolingNhwcMinUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, PoolingNhwcMinUnsignedOp::getRegionBuilder());
      
}

void PoolingNhwcMinUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNhwcMinUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, PoolingNhwcMinUnsignedOp::getRegionBuilder());
  
}

::mlir::LogicalResult PoolingNhwcMinUnsignedOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult PoolingNhwcMinUnsignedOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcMinUnsignedOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcSumOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PoolingNhwcSumOpGenericAdaptorBase::PoolingNhwcSumOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_nhwc_sum", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PoolingNhwcSumOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNhwcSumOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr PoolingNhwcSumOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNhwcSumOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNhwcSumOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &PoolingNhwcSumOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange PoolingNhwcSumOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
PoolingNhwcSumOpAdaptor::PoolingNhwcSumOpAdaptor(PoolingNhwcSumOp op) : PoolingNhwcSumOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PoolingNhwcSumOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.pooling_nhwc_sum' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == PoolingNhwcSumOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == PoolingNhwcSumOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PoolingNhwcSumOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_nhwc_sum' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.pooling_nhwc_sum' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
    return emitError(loc, "'linalg.pooling_nhwc_sum' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNhwcSumOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range PoolingNhwcSumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNhwcSumOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNhwcSumOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNhwcSumOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcSumOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNhwcSumOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNhwcSumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNhwcSumOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNhwcSumOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNhwcSumOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNhwcSumOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute PoolingNhwcSumOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNhwcSumOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, PoolingNhwcSumOp::getRegionBuilder());
      
}

void PoolingNhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, PoolingNhwcSumOp::getRegionBuilder());
      
}

void PoolingNhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, PoolingNhwcSumOp::getRegionBuilder());
  
}

::mlir::LogicalResult PoolingNhwcSumOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult PoolingNhwcSumOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcSumOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNwcMaxOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PoolingNwcMaxOpGenericAdaptorBase::PoolingNwcMaxOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_nwc_max", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PoolingNwcMaxOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNwcMaxOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr PoolingNwcMaxOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNwcMaxOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNwcMaxOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNwcMaxOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNwcMaxOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNwcMaxOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNwcMaxOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &PoolingNwcMaxOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange PoolingNwcMaxOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
PoolingNwcMaxOpAdaptor::PoolingNwcMaxOpAdaptor(PoolingNwcMaxOp op) : PoolingNwcMaxOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PoolingNwcMaxOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.pooling_nwc_max' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == PoolingNwcMaxOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == PoolingNwcMaxOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PoolingNwcMaxOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_nwc_max' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.pooling_nwc_max' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.pooling_nwc_max' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNwcMaxOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range PoolingNwcMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNwcMaxOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNwcMaxOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNwcMaxOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNwcMaxOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNwcMaxOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNwcMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNwcMaxOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNwcMaxOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNwcMaxOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNwcMaxOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNwcMaxOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNwcMaxOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNwcMaxOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNwcMaxOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute PoolingNwcMaxOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNwcMaxOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, PoolingNwcMaxOp::getRegionBuilder());
      
}

void PoolingNwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, PoolingNwcMaxOp::getRegionBuilder());
      
}

void PoolingNwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, PoolingNwcMaxOp::getRegionBuilder());
  
}

::mlir::LogicalResult PoolingNwcMaxOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult PoolingNwcMaxOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNwcMaxOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNwcMaxUnsignedOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PoolingNwcMaxUnsignedOpGenericAdaptorBase::PoolingNwcMaxUnsignedOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_nwc_max_unsigned", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PoolingNwcMaxUnsignedOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNwcMaxUnsignedOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr PoolingNwcMaxUnsignedOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNwcMaxUnsignedOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNwcMaxUnsignedOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNwcMaxUnsignedOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNwcMaxUnsignedOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNwcMaxUnsignedOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNwcMaxUnsignedOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &PoolingNwcMaxUnsignedOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange PoolingNwcMaxUnsignedOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
PoolingNwcMaxUnsignedOpAdaptor::PoolingNwcMaxUnsignedOpAdaptor(PoolingNwcMaxUnsignedOp op) : PoolingNwcMaxUnsignedOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PoolingNwcMaxUnsignedOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.pooling_nwc_max_unsigned' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == PoolingNwcMaxUnsignedOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == PoolingNwcMaxUnsignedOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PoolingNwcMaxUnsignedOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_nwc_max_unsigned' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.pooling_nwc_max_unsigned' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.pooling_nwc_max_unsigned' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNwcMaxUnsignedOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range PoolingNwcMaxUnsignedOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNwcMaxUnsignedOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNwcMaxUnsignedOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNwcMaxUnsignedOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNwcMaxUnsignedOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNwcMaxUnsignedOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNwcMaxUnsignedOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNwcMaxUnsignedOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNwcMaxUnsignedOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNwcMaxUnsignedOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNwcMaxUnsignedOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNwcMaxUnsignedOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNwcMaxUnsignedOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNwcMaxUnsignedOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNwcMaxUnsignedOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute PoolingNwcMaxUnsignedOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNwcMaxUnsignedOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNwcMaxUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, PoolingNwcMaxUnsignedOp::getRegionBuilder());
      
}

void PoolingNwcMaxUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, PoolingNwcMaxUnsignedOp::getRegionBuilder());
      
}

void PoolingNwcMaxUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNwcMaxUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, PoolingNwcMaxUnsignedOp::getRegionBuilder());
  
}

::mlir::LogicalResult PoolingNwcMaxUnsignedOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult PoolingNwcMaxUnsignedOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNwcMaxUnsignedOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNwcMinOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PoolingNwcMinOpGenericAdaptorBase::PoolingNwcMinOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_nwc_min", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PoolingNwcMinOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNwcMinOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr PoolingNwcMinOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNwcMinOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNwcMinOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNwcMinOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNwcMinOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNwcMinOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNwcMinOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &PoolingNwcMinOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange PoolingNwcMinOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
PoolingNwcMinOpAdaptor::PoolingNwcMinOpAdaptor(PoolingNwcMinOp op) : PoolingNwcMinOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PoolingNwcMinOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.pooling_nwc_min' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == PoolingNwcMinOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == PoolingNwcMinOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PoolingNwcMinOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_nwc_min' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.pooling_nwc_min' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.pooling_nwc_min' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNwcMinOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range PoolingNwcMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNwcMinOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNwcMinOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNwcMinOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNwcMinOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNwcMinOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNwcMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNwcMinOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNwcMinOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNwcMinOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNwcMinOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNwcMinOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNwcMinOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNwcMinOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNwcMinOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute PoolingNwcMinOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNwcMinOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, PoolingNwcMinOp::getRegionBuilder());
      
}

void PoolingNwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, PoolingNwcMinOp::getRegionBuilder());
      
}

void PoolingNwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, PoolingNwcMinOp::getRegionBuilder());
  
}

::mlir::LogicalResult PoolingNwcMinOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult PoolingNwcMinOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNwcMinOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNwcMinUnsignedOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PoolingNwcMinUnsignedOpGenericAdaptorBase::PoolingNwcMinUnsignedOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_nwc_min_unsigned", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PoolingNwcMinUnsignedOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNwcMinUnsignedOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr PoolingNwcMinUnsignedOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNwcMinUnsignedOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNwcMinUnsignedOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNwcMinUnsignedOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNwcMinUnsignedOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNwcMinUnsignedOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNwcMinUnsignedOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &PoolingNwcMinUnsignedOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange PoolingNwcMinUnsignedOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
PoolingNwcMinUnsignedOpAdaptor::PoolingNwcMinUnsignedOpAdaptor(PoolingNwcMinUnsignedOp op) : PoolingNwcMinUnsignedOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PoolingNwcMinUnsignedOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.pooling_nwc_min_unsigned' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == PoolingNwcMinUnsignedOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == PoolingNwcMinUnsignedOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PoolingNwcMinUnsignedOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_nwc_min_unsigned' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.pooling_nwc_min_unsigned' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.pooling_nwc_min_unsigned' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNwcMinUnsignedOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range PoolingNwcMinUnsignedOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNwcMinUnsignedOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNwcMinUnsignedOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNwcMinUnsignedOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNwcMinUnsignedOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNwcMinUnsignedOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNwcMinUnsignedOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNwcMinUnsignedOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNwcMinUnsignedOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNwcMinUnsignedOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNwcMinUnsignedOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNwcMinUnsignedOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNwcMinUnsignedOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNwcMinUnsignedOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNwcMinUnsignedOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute PoolingNwcMinUnsignedOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNwcMinUnsignedOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNwcMinUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, PoolingNwcMinUnsignedOp::getRegionBuilder());
      
}

void PoolingNwcMinUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, PoolingNwcMinUnsignedOp::getRegionBuilder());
      
}

void PoolingNwcMinUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNwcMinUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, PoolingNwcMinUnsignedOp::getRegionBuilder());
  
}

::mlir::LogicalResult PoolingNwcMinUnsignedOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult PoolingNwcMinUnsignedOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNwcMinUnsignedOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNwcSumOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PoolingNwcSumOpGenericAdaptorBase::PoolingNwcSumOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_nwc_sum", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PoolingNwcSumOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNwcSumOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr PoolingNwcSumOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNwcSumOpGenericAdaptorBase::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNwcSumOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNwcSumOpGenericAdaptorBase::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNwcSumOpGenericAdaptorBase::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNwcSumOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  if (!attr)
    attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNwcSumOpGenericAdaptorBase::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::Region &PoolingNwcSumOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange PoolingNwcSumOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
PoolingNwcSumOpAdaptor::PoolingNwcSumOpAdaptor(PoolingNwcSumOp op) : PoolingNwcSumOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PoolingNwcSumOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.pooling_nwc_sum' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == PoolingNwcSumOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == PoolingNwcSumOp::getDilationsAttrName(*odsOpName)) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PoolingNwcSumOp::getStridesAttrName(*odsOpName)) {
      tblgen_strides = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_nwc_sum' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.pooling_nwc_sum' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");

  if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
    return emitError(loc, "'linalg.pooling_nwc_sum' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNwcSumOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range PoolingNwcSumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNwcSumOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNwcSumOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNwcSumOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNwcSumOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNwcSumOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNwcSumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNwcSumOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNwcSumOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNwcSumOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNwcSumOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNwcSumOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNwcSumOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::ArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNwcSumOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNwcSumOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

::mlir::Attribute PoolingNwcSumOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNwcSumOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, PoolingNwcSumOp::getRegionBuilder());
      
}

void PoolingNwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, PoolingNwcSumOp::getRegionBuilder());
      
}

void PoolingNwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    buildStructuredOp(odsBuilder, odsState, resultTensorTypes, inputs, outputs,
      attributes, PoolingNwcSumOp::getRegionBuilder());
  
}

::mlir::LogicalResult PoolingNwcSumOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_dilations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDilationsAttrName()) {
      tblgen_dilations = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStridesAttrName()) {
      tblgen_strides = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult PoolingNwcSumOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNwcSumOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::QuantizedBatchMatmulOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
QuantizedBatchMatmulOpGenericAdaptorBase::QuantizedBatchMatmulOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.quantized_batch_matmul", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> QuantizedBatchMatmulOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, QuantizedBatchMatmulOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr QuantizedBatchMatmulOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &QuantizedBatchMatmulOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange QuantizedBatchMatmulOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
QuantizedBatchMatmulOpAdaptor::QuantizedBatchMatmulOpAdaptor(QuantizedBatchMatmulOp op) : QuantizedBatchMatmulOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult QuantizedBatchMatmulOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.quantized_batch_matmul' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == QuantizedBatchMatmulOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.quantized_batch_matmul' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> QuantizedBatchMatmulOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range QuantizedBatchMatmulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range QuantizedBatchMatmulOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range QuantizedBatchMatmulOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange QuantizedBatchMatmulOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange QuantizedBatchMatmulOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> QuantizedBatchMatmulOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range QuantizedBatchMatmulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range QuantizedBatchMatmulOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &QuantizedBatchMatmulOp::getRegion() {
  return (*this)->getRegion(0);
}

void QuantizedBatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, QuantizedBatchMatmulOp::getRegionBuilder());
      
}

void QuantizedBatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, QuantizedBatchMatmulOp::getRegionBuilder());
      
}

void QuantizedBatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult QuantizedBatchMatmulOp::verifyInvariantsImpl() {
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
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult QuantizedBatchMatmulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::QuantizedBatchMatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::QuantizedMatmulOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
QuantizedMatmulOpGenericAdaptorBase::QuantizedMatmulOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.quantized_matmul", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> QuantizedMatmulOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, QuantizedMatmulOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr QuantizedMatmulOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &QuantizedMatmulOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange QuantizedMatmulOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
QuantizedMatmulOpAdaptor::QuantizedMatmulOpAdaptor(QuantizedMatmulOp op) : QuantizedMatmulOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult QuantizedMatmulOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.quantized_matmul' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == QuantizedMatmulOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.quantized_matmul' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> QuantizedMatmulOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range QuantizedMatmulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range QuantizedMatmulOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range QuantizedMatmulOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange QuantizedMatmulOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange QuantizedMatmulOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> QuantizedMatmulOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range QuantizedMatmulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range QuantizedMatmulOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &QuantizedMatmulOp::getRegion() {
  return (*this)->getRegion(0);
}

void QuantizedMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, QuantizedMatmulOp::getRegionBuilder());
      
}

void QuantizedMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, QuantizedMatmulOp::getRegionBuilder());
      
}

void QuantizedMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult QuantizedMatmulOp::verifyInvariantsImpl() {
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
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult QuantizedMatmulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::QuantizedMatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ReduceOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ReduceOpGenericAdaptorBase::ReduceOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.reduce", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ReduceOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 2;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr ReduceOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr ReduceOpGenericAdaptorBase::getDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ReduceOp::getDimensionsAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> ReduceOpGenericAdaptorBase::getDimensions() {
  auto attr = getDimensionsAttr();
  return attr;
}

::mlir::Region &ReduceOpGenericAdaptorBase::getCombiner() {
  return *odsRegions[0];
}

::mlir::RegionRange ReduceOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
ReduceOpAdaptor::ReduceOpAdaptor(ReduceOp op) : ReduceOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ReduceOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.reduce' op ""requires attribute 'dimensions'");
    if (namedAttrIt->getName() == ReduceOp::getDimensionsAttrName(*odsOpName)) {
      tblgen_dimensions = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_dimensions && !(((tblgen_dimensions.isa<::mlir::DenseI64ArrayAttr>())) && (((llvm::is_sorted(tblgen_dimensions.cast<DenseI64ArrayAttr>().asArrayRef()))) && (([](::llvm::ArrayRef<int64_t> a) {
return std::adjacent_find(std::begin(a), std::end(a)) == std::end(a);
}(tblgen_dimensions.cast<DenseI64ArrayAttr>().asArrayRef()))))))
    return emitError(loc, "'linalg.reduce' op ""attribute 'dimensions' failed to satisfy constraint: i64 dense array attribute should be in increasing order");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReduceOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 2;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range ReduceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ReduceOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ReduceOp::getInits() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ReduceOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ReduceOp::getInitsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReduceOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ReduceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &ReduceOp::getCombiner() {
  return (*this)->getRegion(0);
}

::mlir::DenseI64ArrayAttr ReduceOp::getDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getDimensionsAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> ReduceOp::getDimensions() {
  auto attr = getDimensionsAttr();
  return attr;
}

void ReduceOp::setDimensionsAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getDimensionsAttrName(), attr);
}

void ReduceOp::setDimensions(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getDimensionsAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void ReduceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, ::mlir::ValueRange inputs, ::mlir::ValueRange inits, ::mlir::DenseI64ArrayAttr dimensions) {
  odsState.addOperands(inputs);
  odsState.addOperands(inits);
  odsState.addAttribute(getDimensionsAttrName(odsState.name), dimensions);
  (void)odsState.addRegion();
  odsState.addTypes(resultType0);
}

void ReduceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, ::mlir::ValueRange inputs, ::mlir::ValueRange inits, ::llvm::ArrayRef<int64_t> dimensions) {
  odsState.addOperands(inputs);
  odsState.addOperands(inits);
  odsState.addAttribute(getDimensionsAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(dimensions));
  (void)odsState.addRegion();
  odsState.addTypes(resultType0);
}

void ReduceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReduceOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'dimensions'");
    if (namedAttrIt->getName() == getDimensionsAttrName()) {
      tblgen_dimensions = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps10(*this, tblgen_dimensions, "dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps1(*this, region, "combiner", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReduceOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::ReduceOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::TransposeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TransposeOpGenericAdaptorBase::TransposeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.transpose", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TransposeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr TransposeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr TransposeOpGenericAdaptorBase::getPermutationAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TransposeOp::getPermutationAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> TransposeOpGenericAdaptorBase::getPermutation() {
  auto attr = getPermutationAttr();
  return attr;
}

::mlir::Region &TransposeOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange TransposeOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
TransposeOpAdaptor::TransposeOpAdaptor(TransposeOp op) : TransposeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult TransposeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_permutation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.transpose' op ""requires attribute 'permutation'");
    if (namedAttrIt->getName() == TransposeOp::getPermutationAttrName(*odsOpName)) {
      tblgen_permutation = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_permutation && !((tblgen_permutation.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'linalg.transpose' op ""attribute 'permutation' failed to satisfy constraint: i64 dense array attribute");
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

::mlir::TypedValue<::mlir::ShapedType> TransposeOp::getInput() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::ShapedType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::ShapedType> TransposeOp::getInit() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::ShapedType>>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange TransposeOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange TransposeOp::getInitMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TransposeOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range TransposeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range TransposeOp::getResult() {
  return getODSResults(0);
}

::mlir::Region &TransposeOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseI64ArrayAttr TransposeOp::getPermutationAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPermutationAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> TransposeOp::getPermutation() {
  auto attr = getPermutationAttr();
  return attr;
}

void TransposeOp::setPermutationAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getPermutationAttrName(), attr);
}

void TransposeOp::setPermutation(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getPermutationAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps0(*this, tblgen_permutation, "permutation")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult TransposeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::TransposeOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::VecmatOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
VecmatOpGenericAdaptorBase::VecmatOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.vecmat", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> VecmatOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, VecmatOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr VecmatOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &VecmatOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange VecmatOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
VecmatOpAdaptor::VecmatOpAdaptor(VecmatOp op) : VecmatOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult VecmatOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'linalg.vecmat' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == VecmatOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'linalg.vecmat' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> VecmatOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range VecmatOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range VecmatOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range VecmatOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange VecmatOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange VecmatOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> VecmatOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range VecmatOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range VecmatOp::getResultTensors() {
  return getODSResults(0);
}

::mlir::Region &VecmatOp::getRegion() {
  return (*this)->getRegion(0);
}

void VecmatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, std::nullopt, inputs, outputs,
          attributes, VecmatOp::getRegionBuilder());
      
}

void VecmatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, resultTensorTypes,
          inputs, outputs, attributes, VecmatOp::getRegionBuilder());
      
}

void VecmatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult VecmatOp::verifyInvariantsImpl() {
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
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LinalgStructuredOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult VecmatOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::VecmatOp)


#endif  // GET_OP_CLASSES

