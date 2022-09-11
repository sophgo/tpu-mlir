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
::mlir::linalg::BatchMatvecOp,
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
::mlir::linalg::MatmulOp,
::mlir::linalg::MatmulUnsignedOp,
::mlir::linalg::MatvecOp,
::mlir::linalg::Mmt4DOp,
::mlir::linalg::PoolingNchwMaxOp,
::mlir::linalg::PoolingNchwSumOp,
::mlir::linalg::PoolingNdhwcMaxOp,
::mlir::linalg::PoolingNdhwcMinOp,
::mlir::linalg::PoolingNdhwcSumOp,
::mlir::linalg::PoolingNhwcMaxOp,
::mlir::linalg::PoolingNhwcMaxUnsignedOp,
::mlir::linalg::PoolingNhwcMinOp,
::mlir::linalg::PoolingNhwcMinUnsignedOp,
::mlir::linalg::PoolingNhwcSumOp,
::mlir::linalg::QuantizedBatchMatmulOp,
::mlir::linalg::QuantizedMatmulOp,
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
  if (!((((type.isa<::mlir::TensorType>())) && ((type.cast<::mlir::ShapedType>().hasRank()))) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be ranked tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::DenseIntElementsAttr>())) && ((attr.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((attr.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1}))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::DenseIntElementsAttr>())) && ((attr.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((attr.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::DenseIntElementsAttr>())) && ((attr.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((attr.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3}))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::linalg::TypeFnAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::linalg::BinaryFnAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::linalg::UnaryFnAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::AffineMapAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: AffineMap array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps7(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::ArrayAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps8(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::StringAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string attribute";
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
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::BatchMatmulOp definitions
//===----------------------------------------------------------------------===//

BatchMatmulOpAdaptor::BatchMatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.batch_matmul", odsAttrs.getContext());
}

BatchMatmulOpAdaptor::BatchMatmulOpAdaptor(BatchMatmulOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BatchMatmulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BatchMatmulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BatchMatmulOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange BatchMatmulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange BatchMatmulOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange BatchMatmulOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange BatchMatmulOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange BatchMatmulOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr BatchMatmulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange BatchMatmulOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &BatchMatmulOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &BatchMatmulOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range BatchMatmulOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range BatchMatmulOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range BatchMatmulOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange BatchMatmulOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange BatchMatmulOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange BatchMatmulOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange BatchMatmulOp::outputsMutable() {
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

::mlir::Operation::result_range BatchMatmulOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &BatchMatmulOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &BatchMatmulOp::region() {
  return (*this)->getRegion(0);
}

void BatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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
// ::mlir::linalg::BatchMatvecOp definitions
//===----------------------------------------------------------------------===//

BatchMatvecOpAdaptor::BatchMatvecOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.batch_matvec", odsAttrs.getContext());
}

BatchMatvecOpAdaptor::BatchMatvecOpAdaptor(BatchMatvecOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BatchMatvecOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BatchMatvecOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BatchMatvecOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange BatchMatvecOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange BatchMatvecOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange BatchMatvecOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange BatchMatvecOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange BatchMatvecOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr BatchMatvecOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange BatchMatvecOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &BatchMatvecOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &BatchMatvecOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range BatchMatvecOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range BatchMatvecOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range BatchMatvecOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange BatchMatvecOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange BatchMatvecOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange BatchMatvecOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange BatchMatvecOp::outputsMutable() {
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

::mlir::Operation::result_range BatchMatvecOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &BatchMatvecOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &BatchMatvecOp::region() {
  return (*this)->getRegion(0);
}

void BatchMatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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
// ::mlir::linalg::Conv1DNcwFcwOp definitions
//===----------------------------------------------------------------------===//

Conv1DNcwFcwOpAdaptor::Conv1DNcwFcwOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_1d_ncw_fcw", odsAttrs.getContext());
}

Conv1DNcwFcwOpAdaptor::Conv1DNcwFcwOpAdaptor(Conv1DNcwFcwOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange Conv1DNcwFcwOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv1DNcwFcwOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv1DNcwFcwOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange Conv1DNcwFcwOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv1DNcwFcwOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv1DNcwFcwOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv1DNcwFcwOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange Conv1DNcwFcwOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv1DNcwFcwOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv1DNcwFcwOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv1DNcwFcwOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv1DNcwFcwOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv1DNcwFcwOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange Conv1DNcwFcwOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Conv1DNcwFcwOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &Conv1DNcwFcwOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range Conv1DNcwFcwOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv1DNcwFcwOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range Conv1DNcwFcwOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv1DNcwFcwOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv1DNcwFcwOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv1DNcwFcwOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv1DNcwFcwOp::outputsMutable() {
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

::mlir::Operation::result_range Conv1DNcwFcwOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv1DNcwFcwOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &Conv1DNcwFcwOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv1DNcwFcwOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

void Conv1DNcwFcwOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void Conv1DNcwFcwOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void Conv1DNcwFcwOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void Conv1DNcwFcwOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute Conv1DNcwFcwOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute Conv1DNcwFcwOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void Conv1DNcwFcwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void Conv1DNcwFcwOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps0(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps0(*this, tblgen_dilations, "dilations")))
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

Conv1DNwcWcfOpAdaptor::Conv1DNwcWcfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_1d_nwc_wcf", odsAttrs.getContext());
}

Conv1DNwcWcfOpAdaptor::Conv1DNwcWcfOpAdaptor(Conv1DNwcWcfOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange Conv1DNwcWcfOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv1DNwcWcfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv1DNwcWcfOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange Conv1DNwcWcfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv1DNwcWcfOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv1DNwcWcfOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv1DNwcWcfOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange Conv1DNwcWcfOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv1DNwcWcfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv1DNwcWcfOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv1DNwcWcfOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv1DNwcWcfOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv1DNwcWcfOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange Conv1DNwcWcfOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Conv1DNwcWcfOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &Conv1DNwcWcfOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range Conv1DNwcWcfOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv1DNwcWcfOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range Conv1DNwcWcfOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv1DNwcWcfOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv1DNwcWcfOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv1DNwcWcfOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv1DNwcWcfOp::outputsMutable() {
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

::mlir::Operation::result_range Conv1DNwcWcfOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv1DNwcWcfOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &Conv1DNwcWcfOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

void Conv1DNwcWcfOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void Conv1DNwcWcfOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void Conv1DNwcWcfOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void Conv1DNwcWcfOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute Conv1DNwcWcfOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute Conv1DNwcWcfOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void Conv1DNwcWcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void Conv1DNwcWcfOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps0(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps0(*this, tblgen_dilations, "dilations")))
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

Conv1DOpAdaptor::Conv1DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_1d", odsAttrs.getContext());
}

Conv1DOpAdaptor::Conv1DOpAdaptor(Conv1DOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange Conv1DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv1DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv1DOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange Conv1DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv1DOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv1DOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv1DOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange Conv1DOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv1DOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange Conv1DOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Conv1DOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &Conv1DOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range Conv1DOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv1DOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range Conv1DOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv1DOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv1DOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv1DOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv1DOp::outputsMutable() {
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

::mlir::Operation::result_range Conv1DOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv1DOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &Conv1DOp::region() {
  return (*this)->getRegion(0);
}

void Conv1DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

Conv2DNchwFchwOpAdaptor::Conv2DNchwFchwOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_2d_nchw_fchw", odsAttrs.getContext());
}

Conv2DNchwFchwOpAdaptor::Conv2DNchwFchwOpAdaptor(Conv2DNchwFchwOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange Conv2DNchwFchwOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv2DNchwFchwOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv2DNchwFchwOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange Conv2DNchwFchwOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv2DNchwFchwOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv2DNchwFchwOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv2DNchwFchwOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange Conv2DNchwFchwOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv2DNchwFchwOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv2DNchwFchwOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv2DNchwFchwOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv2DNchwFchwOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv2DNchwFchwOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange Conv2DNchwFchwOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Conv2DNchwFchwOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &Conv2DNchwFchwOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range Conv2DNchwFchwOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv2DNchwFchwOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range Conv2DNchwFchwOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv2DNchwFchwOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNchwFchwOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNchwFchwOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNchwFchwOp::outputsMutable() {
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

::mlir::Operation::result_range Conv2DNchwFchwOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv2DNchwFchwOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &Conv2DNchwFchwOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void Conv2DNchwFchwOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void Conv2DNchwFchwOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void Conv2DNchwFchwOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void Conv2DNchwFchwOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute Conv2DNchwFchwOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute Conv2DNchwFchwOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void Conv2DNchwFchwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void Conv2DNchwFchwOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

Conv2DNgchwFgchwOpAdaptor::Conv2DNgchwFgchwOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_2d_ngchw_fgchw", odsAttrs.getContext());
}

Conv2DNgchwFgchwOpAdaptor::Conv2DNgchwFgchwOpAdaptor(Conv2DNgchwFgchwOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange Conv2DNgchwFgchwOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv2DNgchwFgchwOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv2DNgchwFgchwOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange Conv2DNgchwFgchwOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv2DNgchwFgchwOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv2DNgchwFgchwOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv2DNgchwFgchwOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange Conv2DNgchwFgchwOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv2DNgchwFgchwOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv2DNgchwFgchwOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv2DNgchwFgchwOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv2DNgchwFgchwOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv2DNgchwFgchwOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange Conv2DNgchwFgchwOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Conv2DNgchwFgchwOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &Conv2DNgchwFgchwOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range Conv2DNgchwFgchwOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv2DNgchwFgchwOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range Conv2DNgchwFgchwOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv2DNgchwFgchwOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNgchwFgchwOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNgchwFgchwOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNgchwFgchwOp::outputsMutable() {
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

::mlir::Operation::result_range Conv2DNgchwFgchwOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv2DNgchwFgchwOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &Conv2DNgchwFgchwOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNgchwFgchwOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void Conv2DNgchwFgchwOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void Conv2DNgchwFgchwOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void Conv2DNgchwFgchwOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void Conv2DNgchwFgchwOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute Conv2DNgchwFgchwOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute Conv2DNgchwFgchwOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void Conv2DNgchwFgchwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void Conv2DNgchwFgchwOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

Conv2DNhwcFhwcOpAdaptor::Conv2DNhwcFhwcOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_2d_nhwc_fhwc", odsAttrs.getContext());
}

Conv2DNhwcFhwcOpAdaptor::Conv2DNhwcFhwcOpAdaptor(Conv2DNhwcFhwcOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange Conv2DNhwcFhwcOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv2DNhwcFhwcOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv2DNhwcFhwcOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange Conv2DNhwcFhwcOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv2DNhwcFhwcOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv2DNhwcFhwcOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv2DNhwcFhwcOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange Conv2DNhwcFhwcOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv2DNhwcFhwcOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv2DNhwcFhwcOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv2DNhwcFhwcOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv2DNhwcFhwcOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv2DNhwcFhwcOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange Conv2DNhwcFhwcOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Conv2DNhwcFhwcOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &Conv2DNhwcFhwcOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range Conv2DNhwcFhwcOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv2DNhwcFhwcOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range Conv2DNhwcFhwcOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv2DNhwcFhwcOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNhwcFhwcOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNhwcFhwcOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNhwcFhwcOp::outputsMutable() {
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

::mlir::Operation::result_range Conv2DNhwcFhwcOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv2DNhwcFhwcOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &Conv2DNhwcFhwcOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcFhwcOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void Conv2DNhwcFhwcOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void Conv2DNhwcFhwcOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void Conv2DNhwcFhwcOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void Conv2DNhwcFhwcOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute Conv2DNhwcFhwcOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute Conv2DNhwcFhwcOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void Conv2DNhwcFhwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void Conv2DNhwcFhwcOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

Conv2DNhwcHwcfOpAdaptor::Conv2DNhwcHwcfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_2d_nhwc_hwcf", odsAttrs.getContext());
}

Conv2DNhwcHwcfOpAdaptor::Conv2DNhwcHwcfOpAdaptor(Conv2DNhwcHwcfOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange Conv2DNhwcHwcfOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv2DNhwcHwcfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv2DNhwcHwcfOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange Conv2DNhwcHwcfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv2DNhwcHwcfOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv2DNhwcHwcfOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv2DNhwcHwcfOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange Conv2DNhwcHwcfOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv2DNhwcHwcfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv2DNhwcHwcfOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv2DNhwcHwcfOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv2DNhwcHwcfOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv2DNhwcHwcfOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange Conv2DNhwcHwcfOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Conv2DNhwcHwcfOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &Conv2DNhwcHwcfOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range Conv2DNhwcHwcfOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv2DNhwcHwcfOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range Conv2DNhwcHwcfOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv2DNhwcHwcfOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNhwcHwcfOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNhwcHwcfOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNhwcHwcfOp::outputsMutable() {
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

::mlir::Operation::result_range Conv2DNhwcHwcfOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv2DNhwcHwcfOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &Conv2DNhwcHwcfOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void Conv2DNhwcHwcfOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void Conv2DNhwcHwcfOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void Conv2DNhwcHwcfOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void Conv2DNhwcHwcfOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute Conv2DNhwcHwcfOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute Conv2DNhwcHwcfOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void Conv2DNhwcHwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void Conv2DNhwcHwcfOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

Conv2DNhwcHwcfQOpAdaptor::Conv2DNhwcHwcfQOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_2d_nhwc_hwcf_q", odsAttrs.getContext());
}

Conv2DNhwcHwcfQOpAdaptor::Conv2DNhwcHwcfQOpAdaptor(Conv2DNhwcHwcfQOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange Conv2DNhwcHwcfQOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv2DNhwcHwcfQOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv2DNhwcHwcfQOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange Conv2DNhwcHwcfQOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv2DNhwcHwcfQOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv2DNhwcHwcfQOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv2DNhwcHwcfQOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange Conv2DNhwcHwcfQOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv2DNhwcHwcfQOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv2DNhwcHwcfQOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv2DNhwcHwcfQOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv2DNhwcHwcfQOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv2DNhwcHwcfQOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange Conv2DNhwcHwcfQOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Conv2DNhwcHwcfQOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &Conv2DNhwcHwcfQOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range Conv2DNhwcHwcfQOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv2DNhwcHwcfQOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range Conv2DNhwcHwcfQOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv2DNhwcHwcfQOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNhwcHwcfQOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNhwcHwcfQOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNhwcHwcfQOp::outputsMutable() {
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

::mlir::Operation::result_range Conv2DNhwcHwcfQOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv2DNhwcHwcfQOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &Conv2DNhwcHwcfQOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void Conv2DNhwcHwcfQOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void Conv2DNhwcHwcfQOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void Conv2DNhwcHwcfQOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void Conv2DNhwcHwcfQOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute Conv2DNhwcHwcfQOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute Conv2DNhwcHwcfQOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void Conv2DNhwcHwcfQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void Conv2DNhwcHwcfQOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

Conv2DOpAdaptor::Conv2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_2d", odsAttrs.getContext());
}

Conv2DOpAdaptor::Conv2DOpAdaptor(Conv2DOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange Conv2DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv2DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv2DOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange Conv2DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv2DOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv2DOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv2DOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange Conv2DOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv2DOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange Conv2DOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Conv2DOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &Conv2DOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range Conv2DOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv2DOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range Conv2DOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv2DOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DOp::outputsMutable() {
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

::mlir::Operation::result_range Conv2DOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv2DOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &Conv2DOp::region() {
  return (*this)->getRegion(0);
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

Conv3DNdhwcDhwcfOpAdaptor::Conv3DNdhwcDhwcfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_3d_ndhwc_dhwcf", odsAttrs.getContext());
}

Conv3DNdhwcDhwcfOpAdaptor::Conv3DNdhwcDhwcfOpAdaptor(Conv3DNdhwcDhwcfOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange Conv3DNdhwcDhwcfOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv3DNdhwcDhwcfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv3DNdhwcDhwcfOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange Conv3DNdhwcDhwcfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv3DNdhwcDhwcfOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv3DNdhwcDhwcfOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv3DNdhwcDhwcfOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange Conv3DNdhwcDhwcfOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv3DNdhwcDhwcfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv3DNdhwcDhwcfOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, Conv3DNdhwcDhwcfOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv3DNdhwcDhwcfOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, Conv3DNdhwcDhwcfOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange Conv3DNdhwcDhwcfOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Conv3DNdhwcDhwcfOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &Conv3DNdhwcDhwcfOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range Conv3DNdhwcDhwcfOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv3DNdhwcDhwcfOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range Conv3DNdhwcDhwcfOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv3DNdhwcDhwcfOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv3DNdhwcDhwcfOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv3DNdhwcDhwcfOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv3DNdhwcDhwcfOp::outputsMutable() {
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

::mlir::Operation::result_range Conv3DNdhwcDhwcfOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv3DNdhwcDhwcfOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &Conv3DNdhwcDhwcfOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void Conv3DNdhwcDhwcfOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void Conv3DNdhwcDhwcfOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void Conv3DNdhwcDhwcfOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void Conv3DNdhwcDhwcfOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute Conv3DNdhwcDhwcfOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute Conv3DNdhwcDhwcfOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void Conv3DNdhwcDhwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void Conv3DNdhwcDhwcfOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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
// ::mlir::linalg::Conv3DOp definitions
//===----------------------------------------------------------------------===//

Conv3DOpAdaptor::Conv3DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.conv_3d", odsAttrs.getContext());
}

Conv3DOpAdaptor::Conv3DOpAdaptor(Conv3DOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange Conv3DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv3DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Conv3DOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange Conv3DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv3DOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv3DOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv3DOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange Conv3DOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv3DOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange Conv3DOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Conv3DOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &Conv3DOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range Conv3DOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv3DOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range Conv3DOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv3DOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv3DOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv3DOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv3DOp::outputsMutable() {
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

::mlir::Operation::result_range Conv3DOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv3DOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &Conv3DOp::region() {
  return (*this)->getRegion(0);
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

CopyOpAdaptor::CopyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.copy", odsAttrs.getContext());
}

CopyOpAdaptor::CopyOpAdaptor(CopyOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CopyOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CopyOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CopyOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange CopyOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange CopyOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange CopyOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange CopyOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange CopyOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr CopyOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::linalg::TypeFnAttr CopyOpAdaptor::getCastAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CopyOp::getCastAttrName(*odsOpName)).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
    if (!attr)
      attr = ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed);
  return attr;
}

::mlir::linalg::TypeFn CopyOpAdaptor::getCast() {
  auto attr = getCastAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

::mlir::linalg::TypeFnAttr CopyOpAdaptor::castAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CopyOp::getCastAttrName(*odsOpName)).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
    if (!attr)
      attr = ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed);
  return attr;
}

::mlir::linalg::TypeFn CopyOpAdaptor::cast() {
  auto attr = castAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

::mlir::RegionRange CopyOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &CopyOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &CopyOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range CopyOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range CopyOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range CopyOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange CopyOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange CopyOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange CopyOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange CopyOp::outputsMutable() {
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

::mlir::Operation::result_range CopyOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &CopyOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &CopyOp::region() {
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

::mlir::linalg::TypeFnAttr CopyOp::castAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCastAttrName()).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
}

::mlir::linalg::TypeFn CopyOp::cast() {
  auto attr = castAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

void CopyOp::setCastAttr(::mlir::linalg::TypeFnAttr attr) {
  (*this)->setAttr(getCastAttrName(), attr);
}

void CopyOp::castAttr(::mlir::linalg::TypeFnAttr attr) {
  (*this)->setAttr(castAttrName(), attr);
}

::mlir::Attribute CopyOp::removeCastAttr() {
  return (*this)->removeAttr(getCastAttrName());
}

void CopyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void CopyOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps3(*this, tblgen_cast, "cast")))
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

DepthwiseConv1DNwcWcOpAdaptor::DepthwiseConv1DNwcWcOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.depthwise_conv_1d_nwc_wc", odsAttrs.getContext());
}

DepthwiseConv1DNwcWcOpAdaptor::DepthwiseConv1DNwcWcOpAdaptor(DepthwiseConv1DNwcWcOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange DepthwiseConv1DNwcWcOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConv1DNwcWcOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DepthwiseConv1DNwcWcOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange DepthwiseConv1DNwcWcOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DepthwiseConv1DNwcWcOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv1DNwcWcOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv1DNwcWcOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange DepthwiseConv1DNwcWcOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DepthwiseConv1DNwcWcOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv1DNwcWcOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv1DNwcWcOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv1DNwcWcOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv1DNwcWcOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange DepthwiseConv1DNwcWcOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DepthwiseConv1DNwcWcOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &DepthwiseConv1DNwcWcOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range DepthwiseConv1DNwcWcOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv1DNwcWcOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range DepthwiseConv1DNwcWcOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv1DNwcWcOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv1DNwcWcOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv1DNwcWcOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv1DNwcWcOp::outputsMutable() {
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

::mlir::Operation::result_range DepthwiseConv1DNwcWcOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv1DNwcWcOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &DepthwiseConv1DNwcWcOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv1DNwcWcOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void DepthwiseConv1DNwcWcOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void DepthwiseConv1DNwcWcOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void DepthwiseConv1DNwcWcOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute DepthwiseConv1DNwcWcOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute DepthwiseConv1DNwcWcOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void DepthwiseConv1DNwcWcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void DepthwiseConv1DNwcWcOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps0(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps0(*this, tblgen_dilations, "dilations")))
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

DepthwiseConv1DNwcWcmOpAdaptor::DepthwiseConv1DNwcWcmOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.depthwise_conv_1d_nwc_wcm", odsAttrs.getContext());
}

DepthwiseConv1DNwcWcmOpAdaptor::DepthwiseConv1DNwcWcmOpAdaptor(DepthwiseConv1DNwcWcmOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange DepthwiseConv1DNwcWcmOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConv1DNwcWcmOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DepthwiseConv1DNwcWcmOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange DepthwiseConv1DNwcWcmOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DepthwiseConv1DNwcWcmOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv1DNwcWcmOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv1DNwcWcmOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange DepthwiseConv1DNwcWcmOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DepthwiseConv1DNwcWcmOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv1DNwcWcmOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv1DNwcWcmOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv1DNwcWcmOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv1DNwcWcmOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange DepthwiseConv1DNwcWcmOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DepthwiseConv1DNwcWcmOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &DepthwiseConv1DNwcWcmOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range DepthwiseConv1DNwcWcmOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv1DNwcWcmOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range DepthwiseConv1DNwcWcmOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv1DNwcWcmOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv1DNwcWcmOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv1DNwcWcmOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv1DNwcWcmOp::outputsMutable() {
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

::mlir::Operation::result_range DepthwiseConv1DNwcWcmOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv1DNwcWcmOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &DepthwiseConv1DNwcWcmOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcmOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv1DNwcWcmOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void DepthwiseConv1DNwcWcmOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void DepthwiseConv1DNwcWcmOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void DepthwiseConv1DNwcWcmOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute DepthwiseConv1DNwcWcmOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute DepthwiseConv1DNwcWcmOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void DepthwiseConv1DNwcWcmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void DepthwiseConv1DNwcWcmOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps0(*this, tblgen_strides, "strides")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps0(*this, tblgen_dilations, "dilations")))
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

DepthwiseConv2DNchwChwOpAdaptor::DepthwiseConv2DNchwChwOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.depthwise_conv_2d_nchw_chw", odsAttrs.getContext());
}

DepthwiseConv2DNchwChwOpAdaptor::DepthwiseConv2DNchwChwOpAdaptor(DepthwiseConv2DNchwChwOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange DepthwiseConv2DNchwChwOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNchwChwOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DepthwiseConv2DNchwChwOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange DepthwiseConv2DNchwChwOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DepthwiseConv2DNchwChwOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv2DNchwChwOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv2DNchwChwOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange DepthwiseConv2DNchwChwOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DepthwiseConv2DNchwChwOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv2DNchwChwOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv2DNchwChwOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv2DNchwChwOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv2DNchwChwOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange DepthwiseConv2DNchwChwOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DepthwiseConv2DNchwChwOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &DepthwiseConv2DNchwChwOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range DepthwiseConv2DNchwChwOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv2DNchwChwOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range DepthwiseConv2DNchwChwOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv2DNchwChwOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNchwChwOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNchwChwOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNchwChwOp::outputsMutable() {
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

::mlir::Operation::result_range DepthwiseConv2DNchwChwOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv2DNchwChwOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &DepthwiseConv2DNchwChwOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNchwChwOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv2DNchwChwOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void DepthwiseConv2DNchwChwOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void DepthwiseConv2DNchwChwOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void DepthwiseConv2DNchwChwOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute DepthwiseConv2DNchwChwOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute DepthwiseConv2DNchwChwOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void DepthwiseConv2DNchwChwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void DepthwiseConv2DNchwChwOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

DepthwiseConv2DNhwcHwcOpAdaptor::DepthwiseConv2DNhwcHwcOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.depthwise_conv_2d_nhwc_hwc", odsAttrs.getContext());
}

DepthwiseConv2DNhwcHwcOpAdaptor::DepthwiseConv2DNhwcHwcOpAdaptor(DepthwiseConv2DNhwcHwcOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange DepthwiseConv2DNhwcHwcOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DepthwiseConv2DNhwcHwcOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DepthwiseConv2DNhwcHwcOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv2DNhwcHwcOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv2DNhwcHwcOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv2DNhwcHwcOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv2DNhwcHwcOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange DepthwiseConv2DNhwcHwcOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DepthwiseConv2DNhwcHwcOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &DepthwiseConv2DNhwcHwcOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcOp::outputsMutable() {
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

::mlir::Operation::result_range DepthwiseConv2DNhwcHwcOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv2DNhwcHwcOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &DepthwiseConv2DNhwcHwcOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv2DNhwcHwcOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute DepthwiseConv2DNhwcHwcOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute DepthwiseConv2DNhwcHwcOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void DepthwiseConv2DNhwcHwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void DepthwiseConv2DNhwcHwcOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

DepthwiseConv2DNhwcHwcQOpAdaptor::DepthwiseConv2DNhwcHwcQOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.depthwise_conv_2d_nhwc_hwc_q", odsAttrs.getContext());
}

DepthwiseConv2DNhwcHwcQOpAdaptor::DepthwiseConv2DNhwcHwcQOpAdaptor(DepthwiseConv2DNhwcHwcQOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange DepthwiseConv2DNhwcHwcQOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcQOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DepthwiseConv2DNhwcHwcQOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcQOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcQOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcQOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcQOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcQOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DepthwiseConv2DNhwcHwcQOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv2DNhwcHwcQOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv2DNhwcHwcQOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv2DNhwcHwcQOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv2DNhwcHwcQOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange DepthwiseConv2DNhwcHwcQOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DepthwiseConv2DNhwcHwcQOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &DepthwiseConv2DNhwcHwcQOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcQOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcQOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcQOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcQOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcQOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcQOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcQOp::outputsMutable() {
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

::mlir::Operation::result_range DepthwiseConv2DNhwcHwcQOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv2DNhwcHwcQOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &DepthwiseConv2DNhwcHwcQOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv2DNhwcHwcQOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcQOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcQOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcQOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute DepthwiseConv2DNhwcHwcQOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute DepthwiseConv2DNhwcHwcQOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void DepthwiseConv2DNhwcHwcQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void DepthwiseConv2DNhwcHwcQOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

DepthwiseConv2DNhwcHwcmOpAdaptor::DepthwiseConv2DNhwcHwcmOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.depthwise_conv_2d_nhwc_hwcm", odsAttrs.getContext());
}

DepthwiseConv2DNhwcHwcmOpAdaptor::DepthwiseConv2DNhwcHwcmOpAdaptor(DepthwiseConv2DNhwcHwcmOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange DepthwiseConv2DNhwcHwcmOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcmOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DepthwiseConv2DNhwcHwcmOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcmOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcmOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcmOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcmOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcmOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DepthwiseConv2DNhwcHwcmOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv2DNhwcHwcmOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv2DNhwcHwcmOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv2DNhwcHwcmOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv2DNhwcHwcmOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange DepthwiseConv2DNhwcHwcmOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DepthwiseConv2DNhwcHwcmOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &DepthwiseConv2DNhwcHwcmOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcmOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcmOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcmOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcmOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcmOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcmOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcmOp::outputsMutable() {
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

::mlir::Operation::result_range DepthwiseConv2DNhwcHwcmOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv2DNhwcHwcmOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &DepthwiseConv2DNhwcHwcmOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv2DNhwcHwcmOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcmOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcmOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcmOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute DepthwiseConv2DNhwcHwcmOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute DepthwiseConv2DNhwcHwcmOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void DepthwiseConv2DNhwcHwcmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void DepthwiseConv2DNhwcHwcmOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

DepthwiseConv2DNhwcHwcmQOpAdaptor::DepthwiseConv2DNhwcHwcmQOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.depthwise_conv_2d_nhwc_hwcm_q", odsAttrs.getContext());
}

DepthwiseConv2DNhwcHwcmQOpAdaptor::DepthwiseConv2DNhwcHwcmQOpAdaptor(DepthwiseConv2DNhwcHwcmQOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange DepthwiseConv2DNhwcHwcmQOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcmQOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DepthwiseConv2DNhwcHwcmQOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcmQOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcmQOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcmQOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcmQOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcmQOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DepthwiseConv2DNhwcHwcmQOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv2DNhwcHwcmQOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv2DNhwcHwcmQOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv2DNhwcHwcmQOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv2DNhwcHwcmQOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange DepthwiseConv2DNhwcHwcmQOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DepthwiseConv2DNhwcHwcmQOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &DepthwiseConv2DNhwcHwcmQOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcmQOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcmQOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcmQOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcmQOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcmQOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcmQOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcmQOp::outputsMutable() {
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

::mlir::Operation::result_range DepthwiseConv2DNhwcHwcmQOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv2DNhwcHwcmQOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &DepthwiseConv2DNhwcHwcmQOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv2DNhwcHwcmQOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcmQOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcmQOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcmQOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute DepthwiseConv2DNhwcHwcmQOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute DepthwiseConv2DNhwcHwcmQOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void DepthwiseConv2DNhwcHwcmQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void DepthwiseConv2DNhwcHwcmQOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

DepthwiseConv3DNdhwcDhwcOpAdaptor::DepthwiseConv3DNdhwcDhwcOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.depthwise_conv_3d_ndhwc_dhwc", odsAttrs.getContext());
}

DepthwiseConv3DNdhwcDhwcOpAdaptor::DepthwiseConv3DNdhwcDhwcOpAdaptor(DepthwiseConv3DNdhwcDhwcOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange DepthwiseConv3DNdhwcDhwcOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConv3DNdhwcDhwcOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DepthwiseConv3DNdhwcDhwcOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange DepthwiseConv3DNdhwcDhwcOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DepthwiseConv3DNdhwcDhwcOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv3DNdhwcDhwcOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv3DNdhwcDhwcOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange DepthwiseConv3DNdhwcDhwcOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DepthwiseConv3DNdhwcDhwcOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv3DNdhwcDhwcOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv3DNdhwcDhwcOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv3DNdhwcDhwcOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv3DNdhwcDhwcOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange DepthwiseConv3DNdhwcDhwcOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DepthwiseConv3DNdhwcDhwcOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &DepthwiseConv3DNdhwcDhwcOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range DepthwiseConv3DNdhwcDhwcOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv3DNdhwcDhwcOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range DepthwiseConv3DNdhwcDhwcOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv3DNdhwcDhwcOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv3DNdhwcDhwcOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv3DNdhwcDhwcOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv3DNdhwcDhwcOp::outputsMutable() {
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

::mlir::Operation::result_range DepthwiseConv3DNdhwcDhwcOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv3DNdhwcDhwcOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &DepthwiseConv3DNdhwcDhwcOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv3DNdhwcDhwcOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void DepthwiseConv3DNdhwcDhwcOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void DepthwiseConv3DNdhwcDhwcOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void DepthwiseConv3DNdhwcDhwcOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute DepthwiseConv3DNdhwcDhwcOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute DepthwiseConv3DNdhwcDhwcOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void DepthwiseConv3DNdhwcDhwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void DepthwiseConv3DNdhwcDhwcOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

DepthwiseConv3DNdhwcDhwcmOpAdaptor::DepthwiseConv3DNdhwcDhwcmOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.depthwise_conv_3d_ndhwc_dhwcm", odsAttrs.getContext());
}

DepthwiseConv3DNdhwcDhwcmOpAdaptor::DepthwiseConv3DNdhwcDhwcmOpAdaptor(DepthwiseConv3DNdhwcDhwcmOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange DepthwiseConv3DNdhwcDhwcmOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConv3DNdhwcDhwcmOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DepthwiseConv3DNdhwcDhwcmOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange DepthwiseConv3DNdhwcDhwcmOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DepthwiseConv3DNdhwcDhwcmOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv3DNdhwcDhwcmOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv3DNdhwcDhwcmOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange DepthwiseConv3DNdhwcDhwcmOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DepthwiseConv3DNdhwcDhwcmOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv3DNdhwcDhwcmOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DepthwiseConv3DNdhwcDhwcmOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv3DNdhwcDhwcmOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DepthwiseConv3DNdhwcDhwcmOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange DepthwiseConv3DNdhwcDhwcmOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DepthwiseConv3DNdhwcDhwcmOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &DepthwiseConv3DNdhwcDhwcmOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range DepthwiseConv3DNdhwcDhwcmOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv3DNdhwcDhwcmOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range DepthwiseConv3DNdhwcDhwcmOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv3DNdhwcDhwcmOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv3DNdhwcDhwcmOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv3DNdhwcDhwcmOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv3DNdhwcDhwcmOp::outputsMutable() {
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

::mlir::Operation::result_range DepthwiseConv3DNdhwcDhwcmOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv3DNdhwcDhwcmOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &DepthwiseConv3DNdhwcDhwcmOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv3DNdhwcDhwcmOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv3DNdhwcDhwcmOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void DepthwiseConv3DNdhwcDhwcmOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void DepthwiseConv3DNdhwcDhwcmOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void DepthwiseConv3DNdhwcDhwcmOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute DepthwiseConv3DNdhwcDhwcmOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute DepthwiseConv3DNdhwcDhwcmOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void DepthwiseConv3DNdhwcDhwcmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void DepthwiseConv3DNdhwcDhwcmOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

DotOpAdaptor::DotOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.dot", odsAttrs.getContext());
}

DotOpAdaptor::DotOpAdaptor(DotOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange DotOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DotOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DotOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange DotOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DotOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DotOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DotOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange DotOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DotOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange DotOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DotOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &DotOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range DotOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DotOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range DotOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DotOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DotOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DotOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DotOp::outputsMutable() {
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

::mlir::Operation::result_range DotOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DotOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &DotOp::region() {
  return (*this)->getRegion(0);
}

void DotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

ElemwiseBinaryOpAdaptor::ElemwiseBinaryOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.elemwise_binary", odsAttrs.getContext());
}

ElemwiseBinaryOpAdaptor::ElemwiseBinaryOpAdaptor(ElemwiseBinaryOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ElemwiseBinaryOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ElemwiseBinaryOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ElemwiseBinaryOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange ElemwiseBinaryOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ElemwiseBinaryOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ElemwiseBinaryOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ElemwiseBinaryOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange ElemwiseBinaryOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ElemwiseBinaryOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::linalg::BinaryFnAttr ElemwiseBinaryOpAdaptor::getFunAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ElemwiseBinaryOp::getFunAttrName(*odsOpName)).dyn_cast_or_null<::mlir::linalg::BinaryFnAttr>();
    if (!attr)
      attr = ::mlir::linalg::BinaryFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), BinaryFn::add);
  return attr;
}

::mlir::linalg::BinaryFn ElemwiseBinaryOpAdaptor::getFun() {
  auto attr = getFunAttr();
    if (!attr)
      return ::mlir::linalg::BinaryFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), BinaryFn::add).getValue();
  return attr.getValue();
}

::mlir::linalg::BinaryFnAttr ElemwiseBinaryOpAdaptor::funAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ElemwiseBinaryOp::getFunAttrName(*odsOpName)).dyn_cast_or_null<::mlir::linalg::BinaryFnAttr>();
    if (!attr)
      attr = ::mlir::linalg::BinaryFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), BinaryFn::add);
  return attr;
}

::mlir::linalg::BinaryFn ElemwiseBinaryOpAdaptor::fun() {
  auto attr = funAttr();
    if (!attr)
      return ::mlir::linalg::BinaryFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), BinaryFn::add).getValue();
  return attr.getValue();
}

::mlir::linalg::TypeFnAttr ElemwiseBinaryOpAdaptor::getCastAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ElemwiseBinaryOp::getCastAttrName(*odsOpName)).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
    if (!attr)
      attr = ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed);
  return attr;
}

::mlir::linalg::TypeFn ElemwiseBinaryOpAdaptor::getCast() {
  auto attr = getCastAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

::mlir::linalg::TypeFnAttr ElemwiseBinaryOpAdaptor::castAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ElemwiseBinaryOp::getCastAttrName(*odsOpName)).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
    if (!attr)
      attr = ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed);
  return attr;
}

::mlir::linalg::TypeFn ElemwiseBinaryOpAdaptor::cast() {
  auto attr = castAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

::mlir::RegionRange ElemwiseBinaryOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ElemwiseBinaryOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &ElemwiseBinaryOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range ElemwiseBinaryOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ElemwiseBinaryOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range ElemwiseBinaryOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ElemwiseBinaryOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ElemwiseBinaryOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ElemwiseBinaryOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ElemwiseBinaryOp::outputsMutable() {
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

::mlir::Operation::result_range ElemwiseBinaryOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ElemwiseBinaryOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &ElemwiseBinaryOp::region() {
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

::mlir::linalg::BinaryFnAttr ElemwiseBinaryOp::funAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getFunAttrName()).dyn_cast_or_null<::mlir::linalg::BinaryFnAttr>();
}

::mlir::linalg::BinaryFn ElemwiseBinaryOp::fun() {
  auto attr = funAttr();
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

::mlir::linalg::TypeFnAttr ElemwiseBinaryOp::castAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCastAttrName()).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
}

::mlir::linalg::TypeFn ElemwiseBinaryOp::cast() {
  auto attr = castAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

void ElemwiseBinaryOp::setFunAttr(::mlir::linalg::BinaryFnAttr attr) {
  (*this)->setAttr(getFunAttrName(), attr);
}

void ElemwiseBinaryOp::funAttr(::mlir::linalg::BinaryFnAttr attr) {
  (*this)->setAttr(funAttrName(), attr);
}

void ElemwiseBinaryOp::setCastAttr(::mlir::linalg::TypeFnAttr attr) {
  (*this)->setAttr(getCastAttrName(), attr);
}

void ElemwiseBinaryOp::castAttr(::mlir::linalg::TypeFnAttr attr) {
  (*this)->setAttr(castAttrName(), attr);
}

::mlir::Attribute ElemwiseBinaryOp::removeFunAttr() {
  return (*this)->removeAttr(getFunAttrName());
}

::mlir::Attribute ElemwiseBinaryOp::removeCastAttr() {
  return (*this)->removeAttr(getCastAttrName());
}

void ElemwiseBinaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void ElemwiseBinaryOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps4(*this, tblgen_fun, "fun")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps3(*this, tblgen_cast, "cast")))
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

ElemwiseUnaryOpAdaptor::ElemwiseUnaryOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.elemwise_unary", odsAttrs.getContext());
}

ElemwiseUnaryOpAdaptor::ElemwiseUnaryOpAdaptor(ElemwiseUnaryOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ElemwiseUnaryOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ElemwiseUnaryOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ElemwiseUnaryOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange ElemwiseUnaryOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ElemwiseUnaryOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ElemwiseUnaryOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ElemwiseUnaryOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange ElemwiseUnaryOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ElemwiseUnaryOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::linalg::UnaryFnAttr ElemwiseUnaryOpAdaptor::getFunAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ElemwiseUnaryOp::getFunAttrName(*odsOpName)).dyn_cast_or_null<::mlir::linalg::UnaryFnAttr>();
    if (!attr)
      attr = ::mlir::linalg::UnaryFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), UnaryFn::exp);
  return attr;
}

::mlir::linalg::UnaryFn ElemwiseUnaryOpAdaptor::getFun() {
  auto attr = getFunAttr();
    if (!attr)
      return ::mlir::linalg::UnaryFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), UnaryFn::exp).getValue();
  return attr.getValue();
}

::mlir::linalg::UnaryFnAttr ElemwiseUnaryOpAdaptor::funAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ElemwiseUnaryOp::getFunAttrName(*odsOpName)).dyn_cast_or_null<::mlir::linalg::UnaryFnAttr>();
    if (!attr)
      attr = ::mlir::linalg::UnaryFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), UnaryFn::exp);
  return attr;
}

::mlir::linalg::UnaryFn ElemwiseUnaryOpAdaptor::fun() {
  auto attr = funAttr();
    if (!attr)
      return ::mlir::linalg::UnaryFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), UnaryFn::exp).getValue();
  return attr.getValue();
}

::mlir::linalg::TypeFnAttr ElemwiseUnaryOpAdaptor::getCastAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ElemwiseUnaryOp::getCastAttrName(*odsOpName)).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
    if (!attr)
      attr = ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed);
  return attr;
}

::mlir::linalg::TypeFn ElemwiseUnaryOpAdaptor::getCast() {
  auto attr = getCastAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

::mlir::linalg::TypeFnAttr ElemwiseUnaryOpAdaptor::castAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ElemwiseUnaryOp::getCastAttrName(*odsOpName)).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
    if (!attr)
      attr = ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed);
  return attr;
}

::mlir::linalg::TypeFn ElemwiseUnaryOpAdaptor::cast() {
  auto attr = castAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

::mlir::RegionRange ElemwiseUnaryOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ElemwiseUnaryOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &ElemwiseUnaryOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range ElemwiseUnaryOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ElemwiseUnaryOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range ElemwiseUnaryOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ElemwiseUnaryOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ElemwiseUnaryOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ElemwiseUnaryOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ElemwiseUnaryOp::outputsMutable() {
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

::mlir::Operation::result_range ElemwiseUnaryOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ElemwiseUnaryOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &ElemwiseUnaryOp::region() {
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

::mlir::linalg::UnaryFnAttr ElemwiseUnaryOp::funAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getFunAttrName()).dyn_cast_or_null<::mlir::linalg::UnaryFnAttr>();
}

::mlir::linalg::UnaryFn ElemwiseUnaryOp::fun() {
  auto attr = funAttr();
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

::mlir::linalg::TypeFnAttr ElemwiseUnaryOp::castAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCastAttrName()).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
}

::mlir::linalg::TypeFn ElemwiseUnaryOp::cast() {
  auto attr = castAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

void ElemwiseUnaryOp::setFunAttr(::mlir::linalg::UnaryFnAttr attr) {
  (*this)->setAttr(getFunAttrName(), attr);
}

void ElemwiseUnaryOp::funAttr(::mlir::linalg::UnaryFnAttr attr) {
  (*this)->setAttr(funAttrName(), attr);
}

void ElemwiseUnaryOp::setCastAttr(::mlir::linalg::TypeFnAttr attr) {
  (*this)->setAttr(getCastAttrName(), attr);
}

void ElemwiseUnaryOp::castAttr(::mlir::linalg::TypeFnAttr attr) {
  (*this)->setAttr(castAttrName(), attr);
}

::mlir::Attribute ElemwiseUnaryOp::removeFunAttr() {
  return (*this)->removeAttr(getFunAttrName());
}

::mlir::Attribute ElemwiseUnaryOp::removeCastAttr() {
  return (*this)->removeAttr(getCastAttrName());
}

void ElemwiseUnaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void ElemwiseUnaryOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps5(*this, tblgen_fun, "fun")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps3(*this, tblgen_cast, "cast")))
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

FillOpAdaptor::FillOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.fill", odsAttrs.getContext());
}

FillOpAdaptor::FillOpAdaptor(FillOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FillOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FillOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FillOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange FillOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange FillOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange FillOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange FillOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange FillOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr FillOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange FillOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &FillOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &FillOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range FillOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range FillOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range FillOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange FillOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange FillOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange FillOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange FillOp::outputsMutable() {
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

::mlir::Operation::result_range FillOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &FillOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &FillOp::region() {
  return (*this)->getRegion(0);
}

void FillOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

FillRng2DOpAdaptor::FillRng2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.fill_rng_2d", odsAttrs.getContext());
}

FillRng2DOpAdaptor::FillRng2DOpAdaptor(FillRng2DOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FillRng2DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FillRng2DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FillRng2DOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange FillRng2DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange FillRng2DOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange FillRng2DOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange FillRng2DOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange FillRng2DOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr FillRng2DOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange FillRng2DOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &FillRng2DOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &FillRng2DOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range FillRng2DOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range FillRng2DOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range FillRng2DOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange FillRng2DOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange FillRng2DOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange FillRng2DOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange FillRng2DOp::outputsMutable() {
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

::mlir::Operation::result_range FillRng2DOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &FillRng2DOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &FillRng2DOp::region() {
  return (*this)->getRegion(0);
}

void FillRng2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

GenericOpAdaptor::GenericOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.generic", odsAttrs.getContext());
}

GenericOpAdaptor::GenericOpAdaptor(GenericOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange GenericOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GenericOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, GenericOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange GenericOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange GenericOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange GenericOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange GenericOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange GenericOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr GenericOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr GenericOpAdaptor::getIndexingMapsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, GenericOp::getIndexingMapsAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr GenericOpAdaptor::getIndexingMaps() {
  auto attr = getIndexingMapsAttr();
  return attr;
}

::mlir::ArrayAttr GenericOpAdaptor::indexing_mapsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, GenericOp::getIndexingMapsAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr GenericOpAdaptor::indexing_maps() {
  auto attr = indexing_mapsAttr();
  return attr;
}

::mlir::ArrayAttr GenericOpAdaptor::getIteratorTypesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, GenericOp::getIteratorTypesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr GenericOpAdaptor::getIteratorTypes() {
  auto attr = getIteratorTypesAttr();
  return attr;
}

::mlir::ArrayAttr GenericOpAdaptor::iterator_typesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, GenericOp::getIteratorTypesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr GenericOpAdaptor::iterator_types() {
  auto attr = iterator_typesAttr();
  return attr;
}

::mlir::StringAttr GenericOpAdaptor::getDocAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, GenericOp::getDocAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::llvm::Optional< ::llvm::StringRef > GenericOpAdaptor::getDoc() {
  auto attr = getDocAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::StringAttr GenericOpAdaptor::docAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, GenericOp::getDocAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::llvm::Optional< ::llvm::StringRef > GenericOpAdaptor::doc() {
  auto attr = docAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::StringAttr GenericOpAdaptor::getLibraryCallAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, GenericOp::getLibraryCallAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::llvm::Optional< ::llvm::StringRef > GenericOpAdaptor::getLibraryCall() {
  auto attr = getLibraryCallAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::StringAttr GenericOpAdaptor::library_callAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, GenericOp::getLibraryCallAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::llvm::Optional< ::llvm::StringRef > GenericOpAdaptor::library_call() {
  auto attr = library_callAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::RegionRange GenericOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &GenericOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &GenericOpAdaptor::region() {
  return *odsRegions[0];
}

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

  if (tblgen_iterator_types && !((tblgen_iterator_types.isa<::mlir::ArrayAttr>())))
    return emitError(loc, "'linalg.generic' op ""attribute 'iterator_types' failed to satisfy constraint: array attribute");

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

::mlir::Operation::operand_range GenericOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range GenericOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range GenericOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange GenericOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange GenericOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange GenericOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange GenericOp::outputsMutable() {
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

::mlir::Operation::result_range GenericOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &GenericOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &GenericOp::region() {
  return (*this)->getRegion(0);
}

::mlir::ArrayAttr GenericOp::getIndexingMapsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getIndexingMapsAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GenericOp::getIndexingMaps() {
  auto attr = getIndexingMapsAttr();
  return attr;
}

::mlir::ArrayAttr GenericOp::indexing_mapsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getIndexingMapsAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GenericOp::indexing_maps() {
  auto attr = indexing_mapsAttr();
  return attr;
}

::mlir::ArrayAttr GenericOp::getIteratorTypesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getIteratorTypesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GenericOp::getIteratorTypes() {
  auto attr = getIteratorTypesAttr();
  return attr;
}

::mlir::ArrayAttr GenericOp::iterator_typesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getIteratorTypesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GenericOp::iterator_types() {
  auto attr = iterator_typesAttr();
  return attr;
}

::mlir::StringAttr GenericOp::getDocAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getDocAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > GenericOp::getDoc() {
  auto attr = getDocAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::StringAttr GenericOp::docAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getDocAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > GenericOp::doc() {
  auto attr = docAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::StringAttr GenericOp::getLibraryCallAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getLibraryCallAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > GenericOp::getLibraryCall() {
  auto attr = getLibraryCallAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::StringAttr GenericOp::library_callAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getLibraryCallAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > GenericOp::library_call() {
  auto attr = library_callAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void GenericOp::setIndexingMapsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getIndexingMapsAttrName(), attr);
}

void GenericOp::indexing_mapsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(indexing_mapsAttrName(), attr);
}

void GenericOp::setIteratorTypesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getIteratorTypesAttrName(), attr);
}

void GenericOp::iterator_typesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(iterator_typesAttrName(), attr);
}

void GenericOp::setDocAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getDocAttrName(), attr);
}

void GenericOp::docAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(docAttrName(), attr);
}

void GenericOp::setLibraryCallAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getLibraryCallAttrName(), attr);
}

void GenericOp::library_callAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(library_callAttrName(), attr);
}

::mlir::Attribute GenericOp::removeDocAttr() {
  return (*this)->removeAttr(getDocAttrName());
}

::mlir::Attribute GenericOp::removeLibrary_callAttr() {
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps6(*this, tblgen_indexing_maps, "indexing_maps")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps7(*this, tblgen_iterator_types, "iterator_types")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps8(*this, tblgen_doc, "doc")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps8(*this, tblgen_library_call, "library_call")))
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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
// ::mlir::linalg::MatmulOp definitions
//===----------------------------------------------------------------------===//

MatmulOpAdaptor::MatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.matmul", odsAttrs.getContext());
}

MatmulOpAdaptor::MatmulOpAdaptor(MatmulOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MatmulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatmulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MatmulOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange MatmulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MatmulOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatmulOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatmulOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange MatmulOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr MatmulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::linalg::TypeFnAttr MatmulOpAdaptor::getCastAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, MatmulOp::getCastAttrName(*odsOpName)).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
    if (!attr)
      attr = ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed);
  return attr;
}

::mlir::linalg::TypeFn MatmulOpAdaptor::getCast() {
  auto attr = getCastAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

::mlir::linalg::TypeFnAttr MatmulOpAdaptor::castAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, MatmulOp::getCastAttrName(*odsOpName)).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
    if (!attr)
      attr = ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed);
  return attr;
}

::mlir::linalg::TypeFn MatmulOpAdaptor::cast() {
  auto attr = castAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

::mlir::RegionRange MatmulOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &MatmulOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &MatmulOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range MatmulOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatmulOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range MatmulOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatmulOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MatmulOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MatmulOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MatmulOp::outputsMutable() {
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

::mlir::Operation::result_range MatmulOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &MatmulOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &MatmulOp::region() {
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

::mlir::linalg::TypeFnAttr MatmulOp::castAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCastAttrName()).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
}

::mlir::linalg::TypeFn MatmulOp::cast() {
  auto attr = castAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

void MatmulOp::setCastAttr(::mlir::linalg::TypeFnAttr attr) {
  (*this)->setAttr(getCastAttrName(), attr);
}

void MatmulOp::castAttr(::mlir::linalg::TypeFnAttr attr) {
  (*this)->setAttr(castAttrName(), attr);
}

::mlir::Attribute MatmulOp::removeCastAttr() {
  return (*this)->removeAttr(getCastAttrName());
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void MatmulOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps3(*this, tblgen_cast, "cast")))
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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
// ::mlir::linalg::MatmulUnsignedOp definitions
//===----------------------------------------------------------------------===//

MatmulUnsignedOpAdaptor::MatmulUnsignedOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.matmul_unsigned", odsAttrs.getContext());
}

MatmulUnsignedOpAdaptor::MatmulUnsignedOpAdaptor(MatmulUnsignedOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MatmulUnsignedOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatmulUnsignedOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MatmulUnsignedOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange MatmulUnsignedOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MatmulUnsignedOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatmulUnsignedOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatmulUnsignedOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange MatmulUnsignedOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr MatmulUnsignedOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange MatmulUnsignedOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &MatmulUnsignedOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &MatmulUnsignedOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range MatmulUnsignedOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatmulUnsignedOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range MatmulUnsignedOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatmulUnsignedOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MatmulUnsignedOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MatmulUnsignedOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MatmulUnsignedOp::outputsMutable() {
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

::mlir::Operation::result_range MatmulUnsignedOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &MatmulUnsignedOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &MatmulUnsignedOp::region() {
  return (*this)->getRegion(0);
}

void MatmulUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

MatvecOpAdaptor::MatvecOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.matvec", odsAttrs.getContext());
}

MatvecOpAdaptor::MatvecOpAdaptor(MatvecOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MatvecOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatvecOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MatvecOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange MatvecOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MatvecOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatvecOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatvecOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange MatvecOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr MatvecOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange MatvecOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &MatvecOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &MatvecOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range MatvecOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatvecOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range MatvecOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatvecOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MatvecOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MatvecOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MatvecOp::outputsMutable() {
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

::mlir::Operation::result_range MatvecOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &MatvecOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &MatvecOp::region() {
  return (*this)->getRegion(0);
}

void MatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

Mmt4DOpAdaptor::Mmt4DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.mmt4d", odsAttrs.getContext());
}

Mmt4DOpAdaptor::Mmt4DOpAdaptor(Mmt4DOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange Mmt4DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Mmt4DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, Mmt4DOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange Mmt4DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Mmt4DOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Mmt4DOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Mmt4DOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange Mmt4DOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Mmt4DOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange Mmt4DOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Mmt4DOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &Mmt4DOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range Mmt4DOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Mmt4DOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range Mmt4DOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Mmt4DOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Mmt4DOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Mmt4DOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Mmt4DOp::outputsMutable() {
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

::mlir::Operation::result_range Mmt4DOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Mmt4DOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &Mmt4DOp::region() {
  return (*this)->getRegion(0);
}

void Mmt4DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

PoolingNchwMaxOpAdaptor::PoolingNchwMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_nchw_max", odsAttrs.getContext());
}

PoolingNchwMaxOpAdaptor::PoolingNchwMaxOpAdaptor(PoolingNchwMaxOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange PoolingNchwMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNchwMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNchwMaxOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange PoolingNchwMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNchwMaxOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNchwMaxOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNchwMaxOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange PoolingNchwMaxOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNchwMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNchwMaxOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNchwMaxOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNchwMaxOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNchwMaxOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange PoolingNchwMaxOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PoolingNchwMaxOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &PoolingNchwMaxOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range PoolingNchwMaxOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNchwMaxOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range PoolingNchwMaxOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNchwMaxOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNchwMaxOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNchwMaxOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNchwMaxOp::outputsMutable() {
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

::mlir::Operation::result_range PoolingNchwMaxOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNchwMaxOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &PoolingNchwMaxOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNchwMaxOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNchwMaxOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNchwMaxOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void PoolingNchwMaxOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute PoolingNchwMaxOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNchwMaxOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNchwMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void PoolingNchwMaxOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

PoolingNchwSumOpAdaptor::PoolingNchwSumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_nchw_sum", odsAttrs.getContext());
}

PoolingNchwSumOpAdaptor::PoolingNchwSumOpAdaptor(PoolingNchwSumOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange PoolingNchwSumOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNchwSumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNchwSumOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange PoolingNchwSumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNchwSumOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNchwSumOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNchwSumOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange PoolingNchwSumOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNchwSumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNchwSumOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNchwSumOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwSumOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwSumOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNchwSumOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwSumOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwSumOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNchwSumOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwSumOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwSumOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNchwSumOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwSumOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange PoolingNchwSumOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PoolingNchwSumOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &PoolingNchwSumOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range PoolingNchwSumOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNchwSumOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range PoolingNchwSumOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNchwSumOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNchwSumOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNchwSumOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNchwSumOp::outputsMutable() {
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

::mlir::Operation::result_range PoolingNchwSumOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNchwSumOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &PoolingNchwSumOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNchwSumOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNchwSumOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwSumOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNchwSumOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwSumOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNchwSumOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwSumOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNchwSumOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNchwSumOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNchwSumOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNchwSumOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void PoolingNchwSumOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute PoolingNchwSumOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNchwSumOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNchwSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void PoolingNchwSumOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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
// ::mlir::linalg::PoolingNdhwcMaxOp definitions
//===----------------------------------------------------------------------===//

PoolingNdhwcMaxOpAdaptor::PoolingNdhwcMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_ndhwc_max", odsAttrs.getContext());
}

PoolingNdhwcMaxOpAdaptor::PoolingNdhwcMaxOpAdaptor(PoolingNdhwcMaxOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange PoolingNdhwcMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNdhwcMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNdhwcMaxOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange PoolingNdhwcMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNdhwcMaxOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNdhwcMaxOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNdhwcMaxOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange PoolingNdhwcMaxOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNdhwcMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNdhwcMaxOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNdhwcMaxOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNdhwcMaxOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNdhwcMaxOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange PoolingNdhwcMaxOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PoolingNdhwcMaxOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &PoolingNdhwcMaxOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range PoolingNdhwcMaxOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNdhwcMaxOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range PoolingNdhwcMaxOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNdhwcMaxOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNdhwcMaxOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNdhwcMaxOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNdhwcMaxOp::outputsMutable() {
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

::mlir::Operation::result_range PoolingNdhwcMaxOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNdhwcMaxOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &PoolingNdhwcMaxOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNdhwcMaxOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNdhwcMaxOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNdhwcMaxOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void PoolingNdhwcMaxOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute PoolingNdhwcMaxOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNdhwcMaxOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNdhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void PoolingNdhwcMaxOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

PoolingNdhwcMinOpAdaptor::PoolingNdhwcMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_ndhwc_min", odsAttrs.getContext());
}

PoolingNdhwcMinOpAdaptor::PoolingNdhwcMinOpAdaptor(PoolingNdhwcMinOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange PoolingNdhwcMinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNdhwcMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNdhwcMinOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange PoolingNdhwcMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNdhwcMinOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNdhwcMinOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNdhwcMinOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange PoolingNdhwcMinOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNdhwcMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNdhwcMinOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNdhwcMinOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNdhwcMinOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNdhwcMinOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange PoolingNdhwcMinOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PoolingNdhwcMinOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &PoolingNdhwcMinOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range PoolingNdhwcMinOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNdhwcMinOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range PoolingNdhwcMinOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNdhwcMinOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNdhwcMinOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNdhwcMinOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNdhwcMinOp::outputsMutable() {
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

::mlir::Operation::result_range PoolingNdhwcMinOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNdhwcMinOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &PoolingNdhwcMinOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNdhwcMinOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNdhwcMinOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNdhwcMinOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void PoolingNdhwcMinOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute PoolingNdhwcMinOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNdhwcMinOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNdhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void PoolingNdhwcMinOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

PoolingNdhwcSumOpAdaptor::PoolingNdhwcSumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_ndhwc_sum", odsAttrs.getContext());
}

PoolingNdhwcSumOpAdaptor::PoolingNdhwcSumOpAdaptor(PoolingNdhwcSumOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange PoolingNdhwcSumOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNdhwcSumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNdhwcSumOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange PoolingNdhwcSumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNdhwcSumOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNdhwcSumOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNdhwcSumOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange PoolingNdhwcSumOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNdhwcSumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNdhwcSumOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNdhwcSumOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNdhwcSumOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNdhwcSumOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange PoolingNdhwcSumOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PoolingNdhwcSumOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &PoolingNdhwcSumOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range PoolingNdhwcSumOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNdhwcSumOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range PoolingNdhwcSumOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNdhwcSumOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNdhwcSumOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNdhwcSumOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNdhwcSumOp::outputsMutable() {
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

::mlir::Operation::result_range PoolingNdhwcSumOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNdhwcSumOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &PoolingNdhwcSumOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNdhwcSumOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNdhwcSumOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNdhwcSumOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void PoolingNdhwcSumOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute PoolingNdhwcSumOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNdhwcSumOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNdhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void PoolingNdhwcSumOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

PoolingNhwcMaxOpAdaptor::PoolingNhwcMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_nhwc_max", odsAttrs.getContext());
}

PoolingNhwcMaxOpAdaptor::PoolingNhwcMaxOpAdaptor(PoolingNhwcMaxOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange PoolingNhwcMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNhwcMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNhwcMaxOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange PoolingNhwcMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNhwcMaxOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNhwcMaxOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNhwcMaxOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange PoolingNhwcMaxOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNhwcMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNhwcMaxOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNhwcMaxOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNhwcMaxOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNhwcMaxOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange PoolingNhwcMaxOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PoolingNhwcMaxOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &PoolingNhwcMaxOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range PoolingNhwcMaxOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNhwcMaxOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range PoolingNhwcMaxOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNhwcMaxOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMaxOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMaxOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMaxOp::outputsMutable() {
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

::mlir::Operation::result_range PoolingNhwcMaxOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNhwcMaxOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &PoolingNhwcMaxOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNhwcMaxOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNhwcMaxOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNhwcMaxOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void PoolingNhwcMaxOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute PoolingNhwcMaxOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNhwcMaxOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void PoolingNhwcMaxOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

PoolingNhwcMaxUnsignedOpAdaptor::PoolingNhwcMaxUnsignedOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_nhwc_max_unsigned", odsAttrs.getContext());
}

PoolingNhwcMaxUnsignedOpAdaptor::PoolingNhwcMaxUnsignedOpAdaptor(PoolingNhwcMaxUnsignedOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange PoolingNhwcMaxUnsignedOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNhwcMaxUnsignedOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNhwcMaxUnsignedOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange PoolingNhwcMaxUnsignedOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNhwcMaxUnsignedOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNhwcMaxUnsignedOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNhwcMaxUnsignedOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange PoolingNhwcMaxUnsignedOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNhwcMaxUnsignedOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNhwcMaxUnsignedOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNhwcMaxUnsignedOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNhwcMaxUnsignedOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNhwcMaxUnsignedOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange PoolingNhwcMaxUnsignedOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PoolingNhwcMaxUnsignedOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &PoolingNhwcMaxUnsignedOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range PoolingNhwcMaxUnsignedOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNhwcMaxUnsignedOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range PoolingNhwcMaxUnsignedOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNhwcMaxUnsignedOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMaxUnsignedOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMaxUnsignedOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMaxUnsignedOp::outputsMutable() {
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

::mlir::Operation::result_range PoolingNhwcMaxUnsignedOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNhwcMaxUnsignedOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &PoolingNhwcMaxUnsignedOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNhwcMaxUnsignedOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNhwcMaxUnsignedOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNhwcMaxUnsignedOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void PoolingNhwcMaxUnsignedOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute PoolingNhwcMaxUnsignedOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNhwcMaxUnsignedOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNhwcMaxUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void PoolingNhwcMaxUnsignedOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

PoolingNhwcMinOpAdaptor::PoolingNhwcMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_nhwc_min", odsAttrs.getContext());
}

PoolingNhwcMinOpAdaptor::PoolingNhwcMinOpAdaptor(PoolingNhwcMinOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange PoolingNhwcMinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNhwcMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNhwcMinOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange PoolingNhwcMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNhwcMinOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNhwcMinOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNhwcMinOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange PoolingNhwcMinOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNhwcMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNhwcMinOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNhwcMinOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNhwcMinOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNhwcMinOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange PoolingNhwcMinOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PoolingNhwcMinOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &PoolingNhwcMinOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range PoolingNhwcMinOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNhwcMinOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range PoolingNhwcMinOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNhwcMinOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMinOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMinOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMinOp::outputsMutable() {
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

::mlir::Operation::result_range PoolingNhwcMinOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNhwcMinOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &PoolingNhwcMinOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNhwcMinOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNhwcMinOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNhwcMinOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void PoolingNhwcMinOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute PoolingNhwcMinOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNhwcMinOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void PoolingNhwcMinOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

PoolingNhwcMinUnsignedOpAdaptor::PoolingNhwcMinUnsignedOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_nhwc_min_unsigned", odsAttrs.getContext());
}

PoolingNhwcMinUnsignedOpAdaptor::PoolingNhwcMinUnsignedOpAdaptor(PoolingNhwcMinUnsignedOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange PoolingNhwcMinUnsignedOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNhwcMinUnsignedOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNhwcMinUnsignedOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange PoolingNhwcMinUnsignedOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNhwcMinUnsignedOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNhwcMinUnsignedOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNhwcMinUnsignedOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange PoolingNhwcMinUnsignedOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNhwcMinUnsignedOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNhwcMinUnsignedOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNhwcMinUnsignedOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNhwcMinUnsignedOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNhwcMinUnsignedOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange PoolingNhwcMinUnsignedOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PoolingNhwcMinUnsignedOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &PoolingNhwcMinUnsignedOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range PoolingNhwcMinUnsignedOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNhwcMinUnsignedOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range PoolingNhwcMinUnsignedOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNhwcMinUnsignedOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMinUnsignedOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMinUnsignedOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMinUnsignedOp::outputsMutable() {
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

::mlir::Operation::result_range PoolingNhwcMinUnsignedOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNhwcMinUnsignedOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &PoolingNhwcMinUnsignedOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNhwcMinUnsignedOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNhwcMinUnsignedOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNhwcMinUnsignedOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void PoolingNhwcMinUnsignedOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute PoolingNhwcMinUnsignedOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNhwcMinUnsignedOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNhwcMinUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void PoolingNhwcMinUnsignedOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

PoolingNhwcSumOpAdaptor::PoolingNhwcSumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.pooling_nhwc_sum", odsAttrs.getContext());
}

PoolingNhwcSumOpAdaptor::PoolingNhwcSumOpAdaptor(PoolingNhwcSumOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange PoolingNhwcSumOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNhwcSumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PoolingNhwcSumOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange PoolingNhwcSumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNhwcSumOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNhwcSumOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNhwcSumOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange PoolingNhwcSumOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNhwcSumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNhwcSumOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PoolingNhwcSumOp::getStridesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOpAdaptor::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOpAdaptor::getDilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNhwcSumOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOpAdaptor::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, PoolingNhwcSumOp::getDilationsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
    if (!attr)
      attr = ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOpAdaptor::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder(odsAttrs.getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::RegionRange PoolingNhwcSumOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PoolingNhwcSumOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &PoolingNhwcSumOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range PoolingNhwcSumOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNhwcSumOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range PoolingNhwcSumOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNhwcSumOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcSumOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcSumOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcSumOp::outputsMutable() {
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

::mlir::Operation::result_range PoolingNhwcSumOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNhwcSumOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &PoolingNhwcSumOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOp::getStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOp::getStrides() {
  auto attr = getStridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOp::stridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOp::getDilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOp::getDilations() {
  auto attr = getDilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOp::dilationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNhwcSumOp::setStridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void PoolingNhwcSumOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNhwcSumOp::setDilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getDilationsAttrName(), attr);
}

void PoolingNhwcSumOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

::mlir::Attribute PoolingNhwcSumOp::removeStridesAttr() {
  return (*this)->removeAttr(getStridesAttrName());
}

::mlir::Attribute PoolingNhwcSumOp::removeDilationsAttr() {
  return (*this)->removeAttr(getDilationsAttrName());
}

void PoolingNhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

void PoolingNhwcSumOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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
// ::mlir::linalg::QuantizedBatchMatmulOp definitions
//===----------------------------------------------------------------------===//

QuantizedBatchMatmulOpAdaptor::QuantizedBatchMatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.quantized_batch_matmul", odsAttrs.getContext());
}

QuantizedBatchMatmulOpAdaptor::QuantizedBatchMatmulOpAdaptor(QuantizedBatchMatmulOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange QuantizedBatchMatmulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> QuantizedBatchMatmulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, QuantizedBatchMatmulOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange QuantizedBatchMatmulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange QuantizedBatchMatmulOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange QuantizedBatchMatmulOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange QuantizedBatchMatmulOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange QuantizedBatchMatmulOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr QuantizedBatchMatmulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange QuantizedBatchMatmulOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &QuantizedBatchMatmulOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &QuantizedBatchMatmulOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range QuantizedBatchMatmulOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range QuantizedBatchMatmulOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range QuantizedBatchMatmulOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange QuantizedBatchMatmulOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange QuantizedBatchMatmulOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange QuantizedBatchMatmulOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange QuantizedBatchMatmulOp::outputsMutable() {
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

::mlir::Operation::result_range QuantizedBatchMatmulOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &QuantizedBatchMatmulOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &QuantizedBatchMatmulOp::region() {
  return (*this)->getRegion(0);
}

void QuantizedBatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

QuantizedMatmulOpAdaptor::QuantizedMatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.quantized_matmul", odsAttrs.getContext());
}

QuantizedMatmulOpAdaptor::QuantizedMatmulOpAdaptor(QuantizedMatmulOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange QuantizedMatmulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> QuantizedMatmulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, QuantizedMatmulOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange QuantizedMatmulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange QuantizedMatmulOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange QuantizedMatmulOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange QuantizedMatmulOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange QuantizedMatmulOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr QuantizedMatmulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange QuantizedMatmulOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &QuantizedMatmulOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &QuantizedMatmulOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range QuantizedMatmulOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range QuantizedMatmulOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range QuantizedMatmulOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange QuantizedMatmulOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange QuantizedMatmulOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange QuantizedMatmulOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange QuantizedMatmulOp::outputsMutable() {
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

::mlir::Operation::result_range QuantizedMatmulOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &QuantizedMatmulOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &QuantizedMatmulOp::region() {
  return (*this)->getRegion(0);
}

void QuantizedMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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
// ::mlir::linalg::VecmatOp definitions
//===----------------------------------------------------------------------===//

VecmatOpAdaptor::VecmatOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("linalg.vecmat", odsAttrs.getContext());
}

VecmatOpAdaptor::VecmatOpAdaptor(VecmatOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VecmatOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VecmatOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, VecmatOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange VecmatOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange VecmatOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange VecmatOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange VecmatOpAdaptor::getOutputs() {
  return getODSOperands(1);
}

::mlir::ValueRange VecmatOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr VecmatOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange VecmatOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &VecmatOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::Region &VecmatOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::operand_range VecmatOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range VecmatOp::getOutputs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range VecmatOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange VecmatOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange VecmatOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange VecmatOp::getOutputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange VecmatOp::outputsMutable() {
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

::mlir::Operation::result_range VecmatOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &VecmatOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &VecmatOp::region() {
  return (*this)->getRegion(0);
}

void VecmatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        buildStructuredOp(odsBuilder, odsState, llvm::None, inputs, outputs,
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
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

