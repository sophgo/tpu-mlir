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
::mlir::linalg::Conv1DNwcWcfOp,
::mlir::linalg::Conv1DOp,
::mlir::linalg::Conv2DNchwFchwOp,
::mlir::linalg::Conv2DNhwcHwcfOp,
::mlir::linalg::Conv2DNhwcHwcfQOp,
::mlir::linalg::Conv2DOp,
::mlir::linalg::Conv3DNdhwcDhwcfOp,
::mlir::linalg::Conv3DOp,
::mlir::linalg::DepthwiseConv1DNwcWcOp,
::mlir::linalg::DepthwiseConv2DNhwcHwcOp,
::mlir::linalg::DepthwiseConv2DNhwcHwcQOp,
::mlir::linalg::DepthwiseConv2DNhwcHwcmOp,
::mlir::linalg::DepthwiseConv2DNhwcHwcmQOp,
::mlir::linalg::DotOp,
::mlir::linalg::ElemwiseBinaryOp,
::mlir::linalg::ElemwiseUnaryOp,
::mlir::linalg::FillOp,
::mlir::linalg::FillRng2DOp,
::mlir::linalg::FillTensorOp,
::mlir::linalg::GenericOp,
::mlir::linalg::MatmulOp,
::mlir::linalg::MatmulUnsignedOp,
::mlir::linalg::MatvecOp,
::mlir::linalg::Mmt4DOp,
::mlir::linalg::PoolingNchwMaxOp,
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
::mlir::linalg::SoftPlus2DOp,
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

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LinalgStructuredOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::ComplexType>())) || ((type.isa<::mlir::FloatType>())) || ((type.isSignlessInteger())) || ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be complex-type or floating-point or signless integer or vector of any type values, but got " << type;
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
  if (attr && !((attr.isa<::mlir::linalg::BinaryFnAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6";
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
  if (attr && !((attr.isa<::mlir::linalg::UnaryFnAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgStructuredOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::AffineMapAttr>()); })))) {
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

BatchMatmulOpAdaptor::BatchMatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

BatchMatmulOpAdaptor::BatchMatmulOpAdaptor(BatchMatmulOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange BatchMatmulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BatchMatmulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange BatchMatmulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange BatchMatmulOpAdaptor::inputs() {
  return getODSOperands(0);
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

::mlir::Region &BatchMatmulOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult BatchMatmulOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.batch_matmul' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.batch_matmul' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> BatchMatmulOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range BatchMatmulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range BatchMatmulOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range BatchMatmulOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange BatchMatmulOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange BatchMatmulOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range BatchMatmulOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &BatchMatmulOp::region() {
  return (*this)->getRegion(0);
}

void BatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<BatchMatmulOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void BatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<BatchMatmulOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void BatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult BatchMatmulOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
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
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::BatchMatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::BatchMatvecOp definitions
//===----------------------------------------------------------------------===//

BatchMatvecOpAdaptor::BatchMatvecOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

BatchMatvecOpAdaptor::BatchMatvecOpAdaptor(BatchMatvecOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange BatchMatvecOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BatchMatvecOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange BatchMatvecOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange BatchMatvecOpAdaptor::inputs() {
  return getODSOperands(0);
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

::mlir::Region &BatchMatvecOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult BatchMatvecOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.batch_matvec' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.batch_matvec' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> BatchMatvecOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range BatchMatvecOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range BatchMatvecOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range BatchMatvecOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange BatchMatvecOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange BatchMatvecOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range BatchMatvecOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &BatchMatvecOp::region() {
  return (*this)->getRegion(0);
}

void BatchMatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<BatchMatvecOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void BatchMatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<BatchMatvecOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void BatchMatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult BatchMatvecOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
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
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::BatchMatvecOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv1DNwcWcfOp definitions
//===----------------------------------------------------------------------===//

Conv1DNwcWcfOpAdaptor::Conv1DNwcWcfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

Conv1DNwcWcfOpAdaptor::Conv1DNwcWcfOpAdaptor(Conv1DNwcWcfOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange Conv1DNwcWcfOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv1DNwcWcfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange Conv1DNwcWcfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv1DNwcWcfOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv1DNwcWcfOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv1DNwcWcfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::DenseIntElementsAttr Conv1DNwcWcfOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::Region &Conv1DNwcWcfOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult Conv1DNwcWcfOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.conv_1d_nwc_wcf' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.conv_1d_nwc_wcf' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = odsAttrs.get("strides");
    if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
      return emitError(loc, "'linalg.conv_1d_nwc_wcf' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  }
  {
    auto tblgen_dilations = odsAttrs.get("dilations");
    if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
      return emitError(loc, "'linalg.conv_1d_nwc_wcf' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv1DNwcWcfOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range Conv1DNwcWcfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv1DNwcWcfOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv1DNwcWcfOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv1DNwcWcfOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv1DNwcWcfOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range Conv1DNwcWcfOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv1DNwcWcfOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

void Conv1DNwcWcfOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void Conv1DNwcWcfOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void Conv1DNwcWcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv1DNwcWcfOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv1DNwcWcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv1DNwcWcfOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv1DNwcWcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void Conv1DNwcWcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<Conv1DNwcWcfOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult Conv1DNwcWcfOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = (*this)->getAttr(stridesAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps0(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    auto tblgen_dilations = (*this)->getAttr(dilationsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps0(*this, tblgen_dilations, "dilations")))
      return ::mlir::failure();
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

::mlir::LogicalResult Conv1DNwcWcfOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv1DNwcWcfOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv1DOp definitions
//===----------------------------------------------------------------------===//

Conv1DOpAdaptor::Conv1DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

Conv1DOpAdaptor::Conv1DOpAdaptor(Conv1DOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange Conv1DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv1DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange Conv1DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv1DOpAdaptor::inputs() {
  return getODSOperands(0);
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

::mlir::Region &Conv1DOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult Conv1DOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.conv_1d' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.conv_1d' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv1DOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range Conv1DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv1DOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv1DOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv1DOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv1DOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range Conv1DOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv1DOp::region() {
  return (*this)->getRegion(0);
}

void Conv1DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv1DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv1DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv1DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv1DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult Conv1DOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
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
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv1DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DNchwFchwOp definitions
//===----------------------------------------------------------------------===//

Conv2DNchwFchwOpAdaptor::Conv2DNchwFchwOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

Conv2DNchwFchwOpAdaptor::Conv2DNchwFchwOpAdaptor(Conv2DNchwFchwOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange Conv2DNchwFchwOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv2DNchwFchwOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange Conv2DNchwFchwOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv2DNchwFchwOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv2DNchwFchwOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv2DNchwFchwOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::DenseIntElementsAttr Conv2DNchwFchwOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::Region &Conv2DNchwFchwOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult Conv2DNchwFchwOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.conv_2d_nchw_fchw' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.conv_2d_nchw_fchw' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = odsAttrs.get("strides");
    if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.conv_2d_nchw_fchw' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
    auto tblgen_dilations = odsAttrs.get("dilations");
    if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.conv_2d_nchw_fchw' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv2DNchwFchwOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range Conv2DNchwFchwOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv2DNchwFchwOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv2DNchwFchwOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv2DNchwFchwOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNchwFchwOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range Conv2DNchwFchwOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv2DNchwFchwOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void Conv2DNchwFchwOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void Conv2DNchwFchwOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void Conv2DNchwFchwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv2DNchwFchwOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv2DNchwFchwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv2DNchwFchwOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv2DNchwFchwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void Conv2DNchwFchwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<Conv2DNchwFchwOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult Conv2DNchwFchwOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = (*this)->getAttr(stridesAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    auto tblgen_dilations = (*this)->getAttr(dilationsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
      return ::mlir::failure();
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

::mlir::LogicalResult Conv2DNchwFchwOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DNchwFchwOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DNhwcHwcfOp definitions
//===----------------------------------------------------------------------===//

Conv2DNhwcHwcfOpAdaptor::Conv2DNhwcHwcfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

Conv2DNhwcHwcfOpAdaptor::Conv2DNhwcHwcfOpAdaptor(Conv2DNhwcHwcfOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange Conv2DNhwcHwcfOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv2DNhwcHwcfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange Conv2DNhwcHwcfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv2DNhwcHwcfOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv2DNhwcHwcfOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv2DNhwcHwcfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::Region &Conv2DNhwcHwcfOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult Conv2DNhwcHwcfOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.conv_2d_nhwc_hwcf' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.conv_2d_nhwc_hwcf' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = odsAttrs.get("strides");
    if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.conv_2d_nhwc_hwcf' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
    auto tblgen_dilations = odsAttrs.get("dilations");
    if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.conv_2d_nhwc_hwcf' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv2DNhwcHwcfOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range Conv2DNhwcHwcfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv2DNhwcHwcfOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv2DNhwcHwcfOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv2DNhwcHwcfOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNhwcHwcfOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range Conv2DNhwcHwcfOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv2DNhwcHwcfOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void Conv2DNhwcHwcfOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void Conv2DNhwcHwcfOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void Conv2DNhwcHwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv2DNhwcHwcfOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv2DNhwcHwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv2DNhwcHwcfOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv2DNhwcHwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void Conv2DNhwcHwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<Conv2DNhwcHwcfOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult Conv2DNhwcHwcfOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = (*this)->getAttr(stridesAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    auto tblgen_dilations = (*this)->getAttr(dilationsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
      return ::mlir::failure();
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

::mlir::LogicalResult Conv2DNhwcHwcfOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DNhwcHwcfOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DNhwcHwcfQOp definitions
//===----------------------------------------------------------------------===//

Conv2DNhwcHwcfQOpAdaptor::Conv2DNhwcHwcfQOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

Conv2DNhwcHwcfQOpAdaptor::Conv2DNhwcHwcfQOpAdaptor(Conv2DNhwcHwcfQOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange Conv2DNhwcHwcfQOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv2DNhwcHwcfQOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange Conv2DNhwcHwcfQOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv2DNhwcHwcfQOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv2DNhwcHwcfQOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv2DNhwcHwcfQOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::Region &Conv2DNhwcHwcfQOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult Conv2DNhwcHwcfQOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.conv_2d_nhwc_hwcf_q' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.conv_2d_nhwc_hwcf_q' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = odsAttrs.get("strides");
    if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.conv_2d_nhwc_hwcf_q' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
    auto tblgen_dilations = odsAttrs.get("dilations");
    if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.conv_2d_nhwc_hwcf_q' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv2DNhwcHwcfQOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range Conv2DNhwcHwcfQOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv2DNhwcHwcfQOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv2DNhwcHwcfQOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv2DNhwcHwcfQOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNhwcHwcfQOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range Conv2DNhwcHwcfQOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv2DNhwcHwcfQOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void Conv2DNhwcHwcfQOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void Conv2DNhwcHwcfQOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void Conv2DNhwcHwcfQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv2DNhwcHwcfQOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv2DNhwcHwcfQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv2DNhwcHwcfQOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv2DNhwcHwcfQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void Conv2DNhwcHwcfQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<Conv2DNhwcHwcfQOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult Conv2DNhwcHwcfQOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = (*this)->getAttr(stridesAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    auto tblgen_dilations = (*this)->getAttr(dilationsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
      return ::mlir::failure();
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

::mlir::LogicalResult Conv2DNhwcHwcfQOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DNhwcHwcfQOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DOp definitions
//===----------------------------------------------------------------------===//

Conv2DOpAdaptor::Conv2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

Conv2DOpAdaptor::Conv2DOpAdaptor(Conv2DOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange Conv2DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv2DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange Conv2DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv2DOpAdaptor::inputs() {
  return getODSOperands(0);
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

::mlir::Region &Conv2DOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult Conv2DOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.conv_2d' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.conv_2d' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv2DOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range Conv2DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv2DOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv2DOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv2DOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range Conv2DOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv2DOp::region() {
  return (*this)->getRegion(0);
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv2DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv2DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult Conv2DOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
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
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv3DNdhwcDhwcfOp definitions
//===----------------------------------------------------------------------===//

Conv3DNdhwcDhwcfOpAdaptor::Conv3DNdhwcDhwcfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

Conv3DNdhwcDhwcfOpAdaptor::Conv3DNdhwcDhwcfOpAdaptor(Conv3DNdhwcDhwcfOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange Conv3DNdhwcDhwcfOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv3DNdhwcDhwcfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange Conv3DNdhwcDhwcfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv3DNdhwcDhwcfOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv3DNdhwcDhwcfOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv3DNdhwcDhwcfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::Region &Conv3DNdhwcDhwcfOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult Conv3DNdhwcDhwcfOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.conv_3d_ndhwc_dhwcf' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.conv_3d_ndhwc_dhwcf' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = odsAttrs.get("strides");
    if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
      return emitError(loc, "'linalg.conv_3d_ndhwc_dhwcf' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  }
  {
    auto tblgen_dilations = odsAttrs.get("dilations");
    if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
      return emitError(loc, "'linalg.conv_3d_ndhwc_dhwcf' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv3DNdhwcDhwcfOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range Conv3DNdhwcDhwcfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv3DNdhwcDhwcfOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv3DNdhwcDhwcfOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv3DNdhwcDhwcfOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv3DNdhwcDhwcfOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range Conv3DNdhwcDhwcfOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv3DNdhwcDhwcfOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void Conv3DNdhwcDhwcfOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void Conv3DNdhwcDhwcfOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void Conv3DNdhwcDhwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv3DNdhwcDhwcfOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv3DNdhwcDhwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv3DNdhwcDhwcfOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv3DNdhwcDhwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void Conv3DNdhwcDhwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<Conv3DNdhwcDhwcfOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult Conv3DNdhwcDhwcfOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = (*this)->getAttr(stridesAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    auto tblgen_dilations = (*this)->getAttr(dilationsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
      return ::mlir::failure();
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

::mlir::LogicalResult Conv3DNdhwcDhwcfOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv3DNdhwcDhwcfOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv3DOp definitions
//===----------------------------------------------------------------------===//

Conv3DOpAdaptor::Conv3DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

Conv3DOpAdaptor::Conv3DOpAdaptor(Conv3DOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange Conv3DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv3DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange Conv3DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv3DOpAdaptor::inputs() {
  return getODSOperands(0);
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

::mlir::Region &Conv3DOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult Conv3DOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.conv_3d' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.conv_3d' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv3DOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range Conv3DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv3DOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv3DOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv3DOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv3DOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range Conv3DOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv3DOp::region() {
  return (*this)->getRegion(0);
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv3DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv3DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult Conv3DOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
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
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv3DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv1DNwcWcOp definitions
//===----------------------------------------------------------------------===//

DepthwiseConv1DNwcWcOpAdaptor::DepthwiseConv1DNwcWcOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

DepthwiseConv1DNwcWcOpAdaptor::DepthwiseConv1DNwcWcOpAdaptor(DepthwiseConv1DNwcWcOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange DepthwiseConv1DNwcWcOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConv1DNwcWcOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange DepthwiseConv1DNwcWcOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DepthwiseConv1DNwcWcOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv1DNwcWcOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DepthwiseConv1DNwcWcOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::Region &DepthwiseConv1DNwcWcOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult DepthwiseConv1DNwcWcOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.depthwise_conv_1d_nwc_wc' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.depthwise_conv_1d_nwc_wc' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = odsAttrs.get("strides");
    if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
      return emitError(loc, "'linalg.depthwise_conv_1d_nwc_wc' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  }
  {
    auto tblgen_dilations = odsAttrs.get("dilations");
    if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1})))))
      return emitError(loc, "'linalg.depthwise_conv_1d_nwc_wc' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DepthwiseConv1DNwcWcOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range DepthwiseConv1DNwcWcOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DepthwiseConv1DNwcWcOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv1DNwcWcOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv1DNwcWcOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv1DNwcWcOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range DepthwiseConv1DNwcWcOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv1DNwcWcOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv1DNwcWcOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({1}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv1DNwcWcOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void DepthwiseConv1DNwcWcOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void DepthwiseConv1DNwcWcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<DepthwiseConv1DNwcWcOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DepthwiseConv1DNwcWcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConv1DNwcWcOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DepthwiseConv1DNwcWcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void DepthwiseConv1DNwcWcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<DepthwiseConv1DNwcWcOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult DepthwiseConv1DNwcWcOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = (*this)->getAttr(stridesAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps0(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    auto tblgen_dilations = (*this)->getAttr(dilationsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps0(*this, tblgen_dilations, "dilations")))
      return ::mlir::failure();
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

::mlir::LogicalResult DepthwiseConv1DNwcWcOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv1DNwcWcOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNhwcHwcOp definitions
//===----------------------------------------------------------------------===//

DepthwiseConv2DNhwcHwcOpAdaptor::DepthwiseConv2DNhwcHwcOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

DepthwiseConv2DNhwcHwcOpAdaptor::DepthwiseConv2DNhwcHwcOpAdaptor(DepthwiseConv2DNhwcHwcOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange DepthwiseConv2DNhwcHwcOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange DepthwiseConv2DNhwcHwcOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DepthwiseConv2DNhwcHwcOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::Region &DepthwiseConv2DNhwcHwcOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult DepthwiseConv2DNhwcHwcOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwc' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwc' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = odsAttrs.get("strides");
    if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwc' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
    auto tblgen_dilations = odsAttrs.get("dilations");
    if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwc' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range DepthwiseConv2DNhwcHwcOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv2DNhwcHwcOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv2DNhwcHwcOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DepthwiseConv2DNhwcHwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DepthwiseConv2DNhwcHwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void DepthwiseConv2DNhwcHwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult DepthwiseConv2DNhwcHwcOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = (*this)->getAttr(stridesAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    auto tblgen_dilations = (*this)->getAttr(dilationsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
      return ::mlir::failure();
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

::mlir::LogicalResult DepthwiseConv2DNhwcHwcOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNhwcHwcOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNhwcHwcQOp definitions
//===----------------------------------------------------------------------===//

DepthwiseConv2DNhwcHwcQOpAdaptor::DepthwiseConv2DNhwcHwcQOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

DepthwiseConv2DNhwcHwcQOpAdaptor::DepthwiseConv2DNhwcHwcQOpAdaptor(DepthwiseConv2DNhwcHwcQOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange DepthwiseConv2DNhwcHwcQOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcQOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange DepthwiseConv2DNhwcHwcQOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcQOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcQOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DepthwiseConv2DNhwcHwcQOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::Region &DepthwiseConv2DNhwcHwcQOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult DepthwiseConv2DNhwcHwcQOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwc_q' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwc_q' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = odsAttrs.get("strides");
    if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwc_q' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
    auto tblgen_dilations = odsAttrs.get("dilations");
    if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwc_q' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcQOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcQOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcQOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcQOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcQOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcQOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range DepthwiseConv2DNhwcHwcQOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv2DNhwcHwcQOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcQOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv2DNhwcHwcQOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcQOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcQOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DepthwiseConv2DNhwcHwcQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcQOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DepthwiseConv2DNhwcHwcQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void DepthwiseConv2DNhwcHwcQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcQOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult DepthwiseConv2DNhwcHwcQOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = (*this)->getAttr(stridesAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    auto tblgen_dilations = (*this)->getAttr(dilationsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
      return ::mlir::failure();
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

::mlir::LogicalResult DepthwiseConv2DNhwcHwcQOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNhwcHwcQOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNhwcHwcmOp definitions
//===----------------------------------------------------------------------===//

DepthwiseConv2DNhwcHwcmOpAdaptor::DepthwiseConv2DNhwcHwcmOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

DepthwiseConv2DNhwcHwcmOpAdaptor::DepthwiseConv2DNhwcHwcmOpAdaptor(DepthwiseConv2DNhwcHwcmOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange DepthwiseConv2DNhwcHwcmOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcmOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange DepthwiseConv2DNhwcHwcmOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcmOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcmOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DepthwiseConv2DNhwcHwcmOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::Region &DepthwiseConv2DNhwcHwcmOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult DepthwiseConv2DNhwcHwcmOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwcm' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwcm' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = odsAttrs.get("strides");
    if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwcm' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
    auto tblgen_dilations = odsAttrs.get("dilations");
    if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwcm' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcmOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcmOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcmOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcmOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcmOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcmOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range DepthwiseConv2DNhwcHwcmOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv2DNhwcHwcmOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv2DNhwcHwcmOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcmOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcmOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DepthwiseConv2DNhwcHwcmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcmOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DepthwiseConv2DNhwcHwcmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void DepthwiseConv2DNhwcHwcmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcmOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult DepthwiseConv2DNhwcHwcmOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = (*this)->getAttr(stridesAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    auto tblgen_dilations = (*this)->getAttr(dilationsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
      return ::mlir::failure();
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

::mlir::LogicalResult DepthwiseConv2DNhwcHwcmOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNhwcHwcmOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNhwcHwcmQOp definitions
//===----------------------------------------------------------------------===//

DepthwiseConv2DNhwcHwcmQOpAdaptor::DepthwiseConv2DNhwcHwcmQOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

DepthwiseConv2DNhwcHwcmQOpAdaptor::DepthwiseConv2DNhwcHwcmQOpAdaptor(DepthwiseConv2DNhwcHwcmQOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange DepthwiseConv2DNhwcHwcmQOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcmQOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange DepthwiseConv2DNhwcHwcmQOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcmQOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv2DNhwcHwcmQOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DepthwiseConv2DNhwcHwcmQOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::Region &DepthwiseConv2DNhwcHwcmQOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult DepthwiseConv2DNhwcHwcmQOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwcm_q' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwcm_q' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = odsAttrs.get("strides");
    if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwcm_q' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
    auto tblgen_dilations = odsAttrs.get("dilations");
    if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.depthwise_conv_2d_nhwc_hwcm_q' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcHwcmQOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcmQOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcmQOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcHwcmQOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcmQOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcHwcmQOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range DepthwiseConv2DNhwcHwcmQOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv2DNhwcHwcmQOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcHwcmQOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void DepthwiseConv2DNhwcHwcmQOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcmQOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void DepthwiseConv2DNhwcHwcmQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcmQOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DepthwiseConv2DNhwcHwcmQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcmQOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DepthwiseConv2DNhwcHwcmQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void DepthwiseConv2DNhwcHwcmQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcmQOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult DepthwiseConv2DNhwcHwcmQOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = (*this)->getAttr(stridesAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    auto tblgen_dilations = (*this)->getAttr(dilationsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
      return ::mlir::failure();
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

::mlir::LogicalResult DepthwiseConv2DNhwcHwcmQOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNhwcHwcmQOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DotOp definitions
//===----------------------------------------------------------------------===//

DotOpAdaptor::DotOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

DotOpAdaptor::DotOpAdaptor(DotOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange DotOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DotOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange DotOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DotOpAdaptor::inputs() {
  return getODSOperands(0);
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

::mlir::Region &DotOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult DotOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.dot' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.dot' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> DotOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range DotOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DotOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DotOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DotOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DotOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range DotOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DotOp::region() {
  return (*this)->getRegion(0);
}

void DotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<DotOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DotOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult DotOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
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
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DotOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ElemwiseBinaryOp definitions
//===----------------------------------------------------------------------===//

ElemwiseBinaryOpAdaptor::ElemwiseBinaryOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ElemwiseBinaryOpAdaptor::ElemwiseBinaryOpAdaptor(ElemwiseBinaryOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ElemwiseBinaryOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ElemwiseBinaryOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ElemwiseBinaryOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ElemwiseBinaryOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ElemwiseBinaryOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ElemwiseBinaryOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::linalg::BinaryFnAttr ElemwiseBinaryOpAdaptor::funAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::linalg::BinaryFnAttr attr = odsAttrs.get("fun").dyn_cast_or_null<::mlir::linalg::BinaryFnAttr>();
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

::mlir::linalg::TypeFnAttr ElemwiseBinaryOpAdaptor::castAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::linalg::TypeFnAttr attr = odsAttrs.get("cast").dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
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

::mlir::Region &ElemwiseBinaryOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ElemwiseBinaryOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.elemwise_binary' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.elemwise_binary' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_fun = odsAttrs.get("fun");
    if (tblgen_fun && !((tblgen_fun.isa<::mlir::linalg::BinaryFnAttr>())))
      return emitError(loc, "'linalg.elemwise_binary' op ""attribute 'fun' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6");
  }
  {
    auto tblgen_cast = odsAttrs.get("cast");
    if (tblgen_cast && !((tblgen_cast.isa<::mlir::linalg::TypeFnAttr>())))
      return emitError(loc, "'linalg.elemwise_binary' op ""attribute 'cast' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ElemwiseBinaryOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range ElemwiseBinaryOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ElemwiseBinaryOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ElemwiseBinaryOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ElemwiseBinaryOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ElemwiseBinaryOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range ElemwiseBinaryOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ElemwiseBinaryOp::region() {
  return (*this)->getRegion(0);
}

::mlir::linalg::BinaryFnAttr ElemwiseBinaryOp::funAttr() {
  return (*this)->getAttr(funAttrName()).dyn_cast_or_null<::mlir::linalg::BinaryFnAttr>();
}

::mlir::linalg::BinaryFn ElemwiseBinaryOp::fun() {
  auto attr = funAttr();
    if (!attr)
      return ::mlir::linalg::BinaryFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), BinaryFn::add).getValue();
  return attr.getValue();
}

::mlir::linalg::TypeFnAttr ElemwiseBinaryOp::castAttr() {
  return (*this)->getAttr(castAttrName()).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
}

::mlir::linalg::TypeFn ElemwiseBinaryOp::cast() {
  auto attr = castAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

void ElemwiseBinaryOp::funAttr(::mlir::linalg::BinaryFnAttr attr) {
  (*this)->setAttr(funAttrName(), attr);
}

void ElemwiseBinaryOp::castAttr(::mlir::linalg::TypeFnAttr attr) {
  (*this)->setAttr(castAttrName(), attr);
}

void ElemwiseBinaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<ElemwiseBinaryOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void ElemwiseBinaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<ElemwiseBinaryOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void ElemwiseBinaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void ElemwiseBinaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute fun, Attribute cast, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("fun", fun);
odsState.addAttribute("cast", cast);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<ElemwiseBinaryOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult ElemwiseBinaryOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_fun = (*this)->getAttr(funAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps3(*this, tblgen_fun, "fun")))
      return ::mlir::failure();
  }
  {
    auto tblgen_cast = (*this)->getAttr(castAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps4(*this, tblgen_cast, "cast")))
      return ::mlir::failure();
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

::mlir::LogicalResult ElemwiseBinaryOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::ElemwiseBinaryOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ElemwiseUnaryOp definitions
//===----------------------------------------------------------------------===//

ElemwiseUnaryOpAdaptor::ElemwiseUnaryOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ElemwiseUnaryOpAdaptor::ElemwiseUnaryOpAdaptor(ElemwiseUnaryOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ElemwiseUnaryOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ElemwiseUnaryOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ElemwiseUnaryOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ElemwiseUnaryOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ElemwiseUnaryOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ElemwiseUnaryOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::linalg::UnaryFnAttr ElemwiseUnaryOpAdaptor::funAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::linalg::UnaryFnAttr attr = odsAttrs.get("fun").dyn_cast_or_null<::mlir::linalg::UnaryFnAttr>();
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

::mlir::linalg::TypeFnAttr ElemwiseUnaryOpAdaptor::castAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::linalg::TypeFnAttr attr = odsAttrs.get("cast").dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
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

::mlir::Region &ElemwiseUnaryOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ElemwiseUnaryOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.elemwise_unary' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.elemwise_unary' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_fun = odsAttrs.get("fun");
    if (tblgen_fun && !((tblgen_fun.isa<::mlir::linalg::UnaryFnAttr>())))
      return emitError(loc, "'linalg.elemwise_unary' op ""attribute 'fun' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1");
  }
  {
    auto tblgen_cast = odsAttrs.get("cast");
    if (tblgen_cast && !((tblgen_cast.isa<::mlir::linalg::TypeFnAttr>())))
      return emitError(loc, "'linalg.elemwise_unary' op ""attribute 'cast' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ElemwiseUnaryOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range ElemwiseUnaryOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ElemwiseUnaryOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ElemwiseUnaryOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ElemwiseUnaryOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ElemwiseUnaryOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range ElemwiseUnaryOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ElemwiseUnaryOp::region() {
  return (*this)->getRegion(0);
}

::mlir::linalg::UnaryFnAttr ElemwiseUnaryOp::funAttr() {
  return (*this)->getAttr(funAttrName()).dyn_cast_or_null<::mlir::linalg::UnaryFnAttr>();
}

::mlir::linalg::UnaryFn ElemwiseUnaryOp::fun() {
  auto attr = funAttr();
    if (!attr)
      return ::mlir::linalg::UnaryFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), UnaryFn::exp).getValue();
  return attr.getValue();
}

::mlir::linalg::TypeFnAttr ElemwiseUnaryOp::castAttr() {
  return (*this)->getAttr(castAttrName()).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
}

::mlir::linalg::TypeFn ElemwiseUnaryOp::cast() {
  auto attr = castAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

void ElemwiseUnaryOp::funAttr(::mlir::linalg::UnaryFnAttr attr) {
  (*this)->setAttr(funAttrName(), attr);
}

void ElemwiseUnaryOp::castAttr(::mlir::linalg::TypeFnAttr attr) {
  (*this)->setAttr(castAttrName(), attr);
}

void ElemwiseUnaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<ElemwiseUnaryOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void ElemwiseUnaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<ElemwiseUnaryOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void ElemwiseUnaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void ElemwiseUnaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute fun, Attribute cast, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("fun", fun);
odsState.addAttribute("cast", cast);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<ElemwiseUnaryOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult ElemwiseUnaryOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_fun = (*this)->getAttr(funAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps5(*this, tblgen_fun, "fun")))
      return ::mlir::failure();
  }
  {
    auto tblgen_cast = (*this)->getAttr(castAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps4(*this, tblgen_cast, "cast")))
      return ::mlir::failure();
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

::mlir::LogicalResult ElemwiseUnaryOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::ElemwiseUnaryOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::FillOp definitions
//===----------------------------------------------------------------------===//

FillOpAdaptor::FillOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

FillOpAdaptor::FillOpAdaptor(FillOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange FillOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FillOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FillOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FillOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value FillOpAdaptor::output() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr FillOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange FillOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &FillOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult FillOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FillOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FillOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FillOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value FillOp::output() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange FillOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FillOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
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

::mlir::Value FillOp::result() {
  auto results = getODSResults(0);
  return results.empty() ? ::mlir::Value() : *results.begin();
}

::mlir::Region &FillOp::region() {
  return (*this)->getRegion(0);
}

void FillOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type result, ::mlir::Value value, ::mlir::Value output) {
  odsState.addOperands(value);
  odsState.addOperands(output);
  (void)odsState.addRegion();
  if (result)
    odsState.addTypes(result);
}

void FillOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value output) {
  odsState.addOperands(value);
  odsState.addOperands(output);
  (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

void FillOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FillOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(*this, v.getType(), "operand", index++)))
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

    if (valueGroup0.size() > 1) {
      return emitOpError("result group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

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
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult FillOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::OpAsmParser::OperandType outputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> outputOperands(outputRawOperands);  ::llvm::SMLoc outputOperandsLoc;
  (void)outputOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::mlir::Type outputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outputTypes(outputRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> resultTypes;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  if (parser.parseLParen())
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  outputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(outputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
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
    ::mlir::ShapedType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    outputRawTypes[0] = type;
  }
  if (succeeded(parser.parseOptionalArrow())) {

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      resultTypes.push_back(optionalType);
    }
  }
  }
  {
    if (parseFillOpRegion(parser, *regionRegion, valueRawTypes[0], outputRawTypes[0]))
      return ::mlir::failure();
  }
  result.addRegion(std::move(regionRegion));
  result.addTypes(resultTypes);
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(outputOperands, outputTypes, outputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FillOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << value();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << output();
  _odsPrinter << ")";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = value().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = output().getType();
    if (auto validType = type.dyn_cast<::mlir::ShapedType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  if (result()) {
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << (result() ? ::llvm::ArrayRef<::mlir::Type>(result().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  }
  _odsPrinter << ' ';
  printFillOpRegion(_odsPrinter, *this, region(), value().getType(), output().getType());
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::FillOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::FillRng2DOp definitions
//===----------------------------------------------------------------------===//

FillRng2DOpAdaptor::FillRng2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

FillRng2DOpAdaptor::FillRng2DOpAdaptor(FillRng2DOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange FillRng2DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FillRng2DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange FillRng2DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange FillRng2DOpAdaptor::inputs() {
  return getODSOperands(0);
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

::mlir::Region &FillRng2DOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult FillRng2DOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.fill_rng_2d' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.fill_rng_2d' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> FillRng2DOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range FillRng2DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range FillRng2DOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range FillRng2DOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange FillRng2DOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange FillRng2DOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range FillRng2DOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &FillRng2DOp::region() {
  return (*this)->getRegion(0);
}

void FillRng2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<FillRng2DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void FillRng2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<FillRng2DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void FillRng2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult FillRng2DOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
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
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::FillRng2DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::FillTensorOp definitions
//===----------------------------------------------------------------------===//

FillTensorOpAdaptor::FillTensorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

FillTensorOpAdaptor::FillTensorOpAdaptor(FillTensorOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange FillTensorOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FillTensorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange FillTensorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange FillTensorOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange FillTensorOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr FillTensorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange FillTensorOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &FillTensorOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult FillTensorOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.fill_tensor' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.fill_tensor' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> FillTensorOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range FillTensorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range FillTensorOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range FillTensorOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange FillTensorOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange FillTensorOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> FillTensorOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range FillTensorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range FillTensorOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &FillTensorOp::region() {
  return (*this)->getRegion(0);
}

void FillTensorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<FillTensorOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void FillTensorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<FillTensorOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void FillTensorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult FillTensorOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
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

::mlir::LogicalResult FillTensorOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::FillTensorOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::GenericOp definitions
//===----------------------------------------------------------------------===//

GenericOpAdaptor::GenericOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

GenericOpAdaptor::GenericOpAdaptor(GenericOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange GenericOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GenericOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange GenericOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange GenericOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange GenericOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr GenericOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr GenericOpAdaptor::indexing_mapsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("indexing_maps").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr GenericOpAdaptor::indexing_maps() {
  auto attr = indexing_mapsAttr();
  return attr;
}

::mlir::ArrayAttr GenericOpAdaptor::iterator_typesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("iterator_types").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr GenericOpAdaptor::iterator_types() {
  auto attr = iterator_typesAttr();
  return attr;
}

::mlir::StringAttr GenericOpAdaptor::docAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("doc").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::llvm::Optional< ::llvm::StringRef > GenericOpAdaptor::doc() {
  auto attr = docAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::StringAttr GenericOpAdaptor::library_callAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("library_call").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::llvm::Optional< ::llvm::StringRef > GenericOpAdaptor::library_call() {
  auto attr = library_callAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::RegionRange GenericOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &GenericOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult GenericOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.generic' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.generic' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_indexing_maps = odsAttrs.get("indexing_maps");
    if (!tblgen_indexing_maps)
      return emitError(loc, "'linalg.generic' op ""requires attribute 'indexing_maps'");

    if (tblgen_indexing_maps && !(((tblgen_indexing_maps.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_indexing_maps.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::AffineMapAttr>()); }))))
      return emitError(loc, "'linalg.generic' op ""attribute 'indexing_maps' failed to satisfy constraint: AffineMap array attribute");
  }
  {
    auto tblgen_iterator_types = odsAttrs.get("iterator_types");
    if (!tblgen_iterator_types)
      return emitError(loc, "'linalg.generic' op ""requires attribute 'iterator_types'");

    if (tblgen_iterator_types && !((tblgen_iterator_types.isa<::mlir::ArrayAttr>())))
      return emitError(loc, "'linalg.generic' op ""attribute 'iterator_types' failed to satisfy constraint: array attribute");
  }
  {
    auto tblgen_doc = odsAttrs.get("doc");
    if (tblgen_doc && !((tblgen_doc.isa<::mlir::StringAttr>())))
      return emitError(loc, "'linalg.generic' op ""attribute 'doc' failed to satisfy constraint: string attribute");
  }
  {
    auto tblgen_library_call = odsAttrs.get("library_call");
    if (tblgen_library_call && !((tblgen_library_call.isa<::mlir::StringAttr>())))
      return emitError(loc, "'linalg.generic' op ""attribute 'library_call' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GenericOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range GenericOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range GenericOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range GenericOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange GenericOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange GenericOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range GenericOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &GenericOp::region() {
  return (*this)->getRegion(0);
}

::mlir::ArrayAttr GenericOp::indexing_mapsAttr() {
  return (*this)->getAttr(indexing_mapsAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GenericOp::indexing_maps() {
  auto attr = indexing_mapsAttr();
  return attr;
}

::mlir::ArrayAttr GenericOp::iterator_typesAttr() {
  return (*this)->getAttr(iterator_typesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GenericOp::iterator_types() {
  auto attr = iterator_typesAttr();
  return attr;
}

::mlir::StringAttr GenericOp::docAttr() {
  return (*this)->getAttr(docAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > GenericOp::doc() {
  auto attr = docAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::StringAttr GenericOp::library_callAttr() {
  return (*this)->getAttr(library_callAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > GenericOp::library_call() {
  auto attr = library_callAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void GenericOp::indexing_mapsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(indexing_mapsAttrName(), attr);
}

void GenericOp::iterator_typesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(iterator_typesAttrName(), attr);
}

void GenericOp::docAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(docAttrName(), attr);
}

void GenericOp::library_callAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(library_callAttrName(), attr);
}

::mlir::Attribute GenericOp::removeDocAttr() {
  return (*this)->removeAttr(docAttrName());
}

::mlir::Attribute GenericOp::removeLibrary_callAttr() {
  return (*this)->removeAttr(library_callAttrName());
}

void GenericOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result_tensors, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::ArrayAttr indexing_maps, ::mlir::ArrayAttr iterator_types, /*optional*/::mlir::StringAttr doc, /*optional*/::mlir::StringAttr library_call) {
  odsState.addOperands(inputs);
  odsState.addOperands(outputs);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(outputs.size())}));
  odsState.addAttribute(indexing_mapsAttrName(odsState.name), indexing_maps);
  odsState.addAttribute(iterator_typesAttrName(odsState.name), iterator_types);
  if (doc) {
  odsState.addAttribute(docAttrName(odsState.name), doc);
  }
  if (library_call) {
  odsState.addAttribute(library_callAttrName(odsState.name), library_call);
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
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_indexing_maps = (*this)->getAttr(indexing_mapsAttrName());
    if (!tblgen_indexing_maps)
      return emitOpError("requires attribute 'indexing_maps'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps6(*this, tblgen_indexing_maps, "indexing_maps")))
      return ::mlir::failure();
  }
  {
    auto tblgen_iterator_types = (*this)->getAttr(iterator_typesAttrName());
    if (!tblgen_iterator_types)
      return emitOpError("requires attribute 'iterator_types'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps7(*this, tblgen_iterator_types, "iterator_types")))
      return ::mlir::failure();
  }
  {
    auto tblgen_doc = (*this)->getAttr(docAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps8(*this, tblgen_doc, "doc")))
      return ::mlir::failure();
  }
  {
    auto tblgen_library_call = (*this)->getAttr(library_callAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps8(*this, tblgen_library_call, "library_call")))
      return ::mlir::failure();
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

::mlir::LogicalResult GenericOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::GenericOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatmulOp definitions
//===----------------------------------------------------------------------===//

MatmulOpAdaptor::MatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

MatmulOpAdaptor::MatmulOpAdaptor(MatmulOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange MatmulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatmulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange MatmulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MatmulOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatmulOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr MatmulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::linalg::TypeFnAttr MatmulOpAdaptor::castAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::linalg::TypeFnAttr attr = odsAttrs.get("cast").dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
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

::mlir::Region &MatmulOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult MatmulOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.matmul' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.matmul' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_cast = odsAttrs.get("cast");
    if (tblgen_cast && !((tblgen_cast.isa<::mlir::linalg::TypeFnAttr>())))
      return emitError(loc, "'linalg.matmul' op ""attribute 'cast' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MatmulOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range MatmulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatmulOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatmulOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatmulOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MatmulOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range MatmulOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &MatmulOp::region() {
  return (*this)->getRegion(0);
}

::mlir::linalg::TypeFnAttr MatmulOp::castAttr() {
  return (*this)->getAttr(castAttrName()).dyn_cast_or_null<::mlir::linalg::TypeFnAttr>();
}

::mlir::linalg::TypeFn MatmulOp::cast() {
  auto attr = castAttr();
    if (!attr)
      return ::mlir::linalg::TypeFnAttr::get(::mlir::Builder((*this)->getContext()).getContext(), TypeFn::cast_signed).getValue();
  return attr.getValue();
}

void MatmulOp::castAttr(::mlir::linalg::TypeFnAttr attr) {
  (*this)->setAttr(castAttrName(), attr);
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<MatmulOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<MatmulOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute cast, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("cast", cast);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<MatmulOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult MatmulOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_cast = (*this)->getAttr(castAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps4(*this, tblgen_cast, "cast")))
      return ::mlir::failure();
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

::mlir::LogicalResult MatmulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::MatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatmulUnsignedOp definitions
//===----------------------------------------------------------------------===//

MatmulUnsignedOpAdaptor::MatmulUnsignedOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

MatmulUnsignedOpAdaptor::MatmulUnsignedOpAdaptor(MatmulUnsignedOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange MatmulUnsignedOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatmulUnsignedOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange MatmulUnsignedOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MatmulUnsignedOpAdaptor::inputs() {
  return getODSOperands(0);
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

::mlir::Region &MatmulUnsignedOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult MatmulUnsignedOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.matmul_unsigned' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.matmul_unsigned' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> MatmulUnsignedOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range MatmulUnsignedOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatmulUnsignedOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatmulUnsignedOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatmulUnsignedOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MatmulUnsignedOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range MatmulUnsignedOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &MatmulUnsignedOp::region() {
  return (*this)->getRegion(0);
}

void MatmulUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<MatmulUnsignedOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void MatmulUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<MatmulUnsignedOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void MatmulUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult MatmulUnsignedOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
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
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::MatmulUnsignedOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatvecOp definitions
//===----------------------------------------------------------------------===//

MatvecOpAdaptor::MatvecOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

MatvecOpAdaptor::MatvecOpAdaptor(MatvecOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange MatvecOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatvecOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange MatvecOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MatvecOpAdaptor::inputs() {
  return getODSOperands(0);
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

::mlir::Region &MatvecOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult MatvecOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.matvec' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.matvec' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> MatvecOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range MatvecOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatvecOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatvecOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatvecOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MatvecOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range MatvecOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &MatvecOp::region() {
  return (*this)->getRegion(0);
}

void MatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<MatvecOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void MatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<MatvecOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void MatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult MatvecOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
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
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::MatvecOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Mmt4DOp definitions
//===----------------------------------------------------------------------===//

Mmt4DOpAdaptor::Mmt4DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

Mmt4DOpAdaptor::Mmt4DOpAdaptor(Mmt4DOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange Mmt4DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Mmt4DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange Mmt4DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Mmt4DOpAdaptor::inputs() {
  return getODSOperands(0);
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

::mlir::Region &Mmt4DOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult Mmt4DOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.mmt4d' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.mmt4d' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> Mmt4DOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range Mmt4DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Mmt4DOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Mmt4DOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Mmt4DOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Mmt4DOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range Mmt4DOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Mmt4DOp::region() {
  return (*this)->getRegion(0);
}

void Mmt4DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<Mmt4DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Mmt4DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Mmt4DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Mmt4DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult Mmt4DOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
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
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Mmt4DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNchwMaxOp definitions
//===----------------------------------------------------------------------===//

PoolingNchwMaxOpAdaptor::PoolingNchwMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

PoolingNchwMaxOpAdaptor::PoolingNchwMaxOpAdaptor(PoolingNchwMaxOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange PoolingNchwMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNchwMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange PoolingNchwMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNchwMaxOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNchwMaxOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNchwMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::DenseIntElementsAttr PoolingNchwMaxOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::Region &PoolingNchwMaxOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult PoolingNchwMaxOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.pooling_nchw_max' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_nchw_max' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = odsAttrs.get("strides");
    if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.pooling_nchw_max' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
    auto tblgen_dilations = odsAttrs.get("dilations");
    if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.pooling_nchw_max' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNchwMaxOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range PoolingNchwMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNchwMaxOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNchwMaxOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNchwMaxOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNchwMaxOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range PoolingNchwMaxOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNchwMaxOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNchwMaxOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNchwMaxOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void PoolingNchwMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNchwMaxOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNchwMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNchwMaxOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNchwMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNchwMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNchwMaxOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult PoolingNchwMaxOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = (*this)->getAttr(stridesAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    auto tblgen_dilations = (*this)->getAttr(dilationsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
      return ::mlir::failure();
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

::mlir::LogicalResult PoolingNchwMaxOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNchwMaxOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNdhwcMaxOp definitions
//===----------------------------------------------------------------------===//

PoolingNdhwcMaxOpAdaptor::PoolingNdhwcMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

PoolingNdhwcMaxOpAdaptor::PoolingNdhwcMaxOpAdaptor(PoolingNdhwcMaxOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange PoolingNdhwcMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNdhwcMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange PoolingNdhwcMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNdhwcMaxOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNdhwcMaxOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNdhwcMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::Region &PoolingNdhwcMaxOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult PoolingNdhwcMaxOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.pooling_ndhwc_max' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_ndhwc_max' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = odsAttrs.get("strides");
    if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
      return emitError(loc, "'linalg.pooling_ndhwc_max' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  }
  {
    auto tblgen_dilations = odsAttrs.get("dilations");
    if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
      return emitError(loc, "'linalg.pooling_ndhwc_max' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNdhwcMaxOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range PoolingNdhwcMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNdhwcMaxOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNdhwcMaxOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNdhwcMaxOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNdhwcMaxOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range PoolingNdhwcMaxOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNdhwcMaxOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNdhwcMaxOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNdhwcMaxOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void PoolingNdhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNdhwcMaxOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNdhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNdhwcMaxOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNdhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNdhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNdhwcMaxOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult PoolingNdhwcMaxOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = (*this)->getAttr(stridesAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    auto tblgen_dilations = (*this)->getAttr(dilationsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
      return ::mlir::failure();
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

::mlir::LogicalResult PoolingNdhwcMaxOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNdhwcMaxOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNdhwcMinOp definitions
//===----------------------------------------------------------------------===//

PoolingNdhwcMinOpAdaptor::PoolingNdhwcMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

PoolingNdhwcMinOpAdaptor::PoolingNdhwcMinOpAdaptor(PoolingNdhwcMinOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange PoolingNdhwcMinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNdhwcMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange PoolingNdhwcMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNdhwcMinOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNdhwcMinOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNdhwcMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::DenseIntElementsAttr PoolingNdhwcMinOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::Region &PoolingNdhwcMinOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult PoolingNdhwcMinOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.pooling_ndhwc_min' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_ndhwc_min' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = odsAttrs.get("strides");
    if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
      return emitError(loc, "'linalg.pooling_ndhwc_min' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  }
  {
    auto tblgen_dilations = odsAttrs.get("dilations");
    if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
      return emitError(loc, "'linalg.pooling_ndhwc_min' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNdhwcMinOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range PoolingNdhwcMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNdhwcMinOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNdhwcMinOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNdhwcMinOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNdhwcMinOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range PoolingNdhwcMinOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNdhwcMinOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNdhwcMinOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNdhwcMinOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void PoolingNdhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNdhwcMinOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNdhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNdhwcMinOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNdhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNdhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNdhwcMinOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult PoolingNdhwcMinOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = (*this)->getAttr(stridesAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    auto tblgen_dilations = (*this)->getAttr(dilationsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
      return ::mlir::failure();
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

::mlir::LogicalResult PoolingNdhwcMinOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNdhwcMinOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNdhwcSumOp definitions
//===----------------------------------------------------------------------===//

PoolingNdhwcSumOpAdaptor::PoolingNdhwcSumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

PoolingNdhwcSumOpAdaptor::PoolingNdhwcSumOpAdaptor(PoolingNdhwcSumOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange PoolingNdhwcSumOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNdhwcSumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange PoolingNdhwcSumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNdhwcSumOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNdhwcSumOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNdhwcSumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::DenseIntElementsAttr PoolingNdhwcSumOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::Region &PoolingNdhwcSumOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult PoolingNdhwcSumOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.pooling_ndhwc_sum' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_ndhwc_sum' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = odsAttrs.get("strides");
    if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
      return emitError(loc, "'linalg.pooling_ndhwc_sum' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  }
  {
    auto tblgen_dilations = odsAttrs.get("dilations");
    if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3})))))
      return emitError(loc, "'linalg.pooling_ndhwc_sum' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNdhwcSumOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range PoolingNdhwcSumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNdhwcSumOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNdhwcSumOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNdhwcSumOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNdhwcSumOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range PoolingNdhwcSumOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNdhwcSumOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({3}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNdhwcSumOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNdhwcSumOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void PoolingNdhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNdhwcSumOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNdhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNdhwcSumOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNdhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNdhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNdhwcSumOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult PoolingNdhwcSumOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = (*this)->getAttr(stridesAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    auto tblgen_dilations = (*this)->getAttr(dilationsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps2(*this, tblgen_dilations, "dilations")))
      return ::mlir::failure();
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

::mlir::LogicalResult PoolingNdhwcSumOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNdhwcSumOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcMaxOp definitions
//===----------------------------------------------------------------------===//

PoolingNhwcMaxOpAdaptor::PoolingNhwcMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

PoolingNhwcMaxOpAdaptor::PoolingNhwcMaxOpAdaptor(PoolingNhwcMaxOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange PoolingNhwcMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNhwcMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange PoolingNhwcMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNhwcMaxOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNhwcMaxOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNhwcMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::DenseIntElementsAttr PoolingNhwcMaxOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::Region &PoolingNhwcMaxOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult PoolingNhwcMaxOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.pooling_nhwc_max' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_nhwc_max' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = odsAttrs.get("strides");
    if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.pooling_nhwc_max' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
    auto tblgen_dilations = odsAttrs.get("dilations");
    if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.pooling_nhwc_max' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNhwcMaxOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range PoolingNhwcMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNhwcMaxOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNhwcMaxOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNhwcMaxOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMaxOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range PoolingNhwcMaxOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNhwcMaxOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNhwcMaxOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNhwcMaxOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void PoolingNhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNhwcMaxOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNhwcMaxOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNhwcMaxOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult PoolingNhwcMaxOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = (*this)->getAttr(stridesAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    auto tblgen_dilations = (*this)->getAttr(dilationsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
      return ::mlir::failure();
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

::mlir::LogicalResult PoolingNhwcMaxOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcMaxOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcMaxUnsignedOp definitions
//===----------------------------------------------------------------------===//

PoolingNhwcMaxUnsignedOpAdaptor::PoolingNhwcMaxUnsignedOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

PoolingNhwcMaxUnsignedOpAdaptor::PoolingNhwcMaxUnsignedOpAdaptor(PoolingNhwcMaxUnsignedOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange PoolingNhwcMaxUnsignedOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNhwcMaxUnsignedOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange PoolingNhwcMaxUnsignedOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNhwcMaxUnsignedOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNhwcMaxUnsignedOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNhwcMaxUnsignedOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::Region &PoolingNhwcMaxUnsignedOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult PoolingNhwcMaxUnsignedOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.pooling_nhwc_max_unsigned' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_nhwc_max_unsigned' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = odsAttrs.get("strides");
    if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.pooling_nhwc_max_unsigned' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
    auto tblgen_dilations = odsAttrs.get("dilations");
    if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.pooling_nhwc_max_unsigned' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNhwcMaxUnsignedOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range PoolingNhwcMaxUnsignedOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNhwcMaxUnsignedOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNhwcMaxUnsignedOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNhwcMaxUnsignedOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMaxUnsignedOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range PoolingNhwcMaxUnsignedOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNhwcMaxUnsignedOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxUnsignedOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNhwcMaxUnsignedOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNhwcMaxUnsignedOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void PoolingNhwcMaxUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNhwcMaxUnsignedOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNhwcMaxUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNhwcMaxUnsignedOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNhwcMaxUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNhwcMaxUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNhwcMaxUnsignedOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult PoolingNhwcMaxUnsignedOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = (*this)->getAttr(stridesAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    auto tblgen_dilations = (*this)->getAttr(dilationsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
      return ::mlir::failure();
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

::mlir::LogicalResult PoolingNhwcMaxUnsignedOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcMaxUnsignedOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcMinOp definitions
//===----------------------------------------------------------------------===//

PoolingNhwcMinOpAdaptor::PoolingNhwcMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

PoolingNhwcMinOpAdaptor::PoolingNhwcMinOpAdaptor(PoolingNhwcMinOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange PoolingNhwcMinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNhwcMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange PoolingNhwcMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNhwcMinOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNhwcMinOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNhwcMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::DenseIntElementsAttr PoolingNhwcMinOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::Region &PoolingNhwcMinOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult PoolingNhwcMinOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.pooling_nhwc_min' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_nhwc_min' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = odsAttrs.get("strides");
    if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.pooling_nhwc_min' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
    auto tblgen_dilations = odsAttrs.get("dilations");
    if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.pooling_nhwc_min' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNhwcMinOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range PoolingNhwcMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNhwcMinOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNhwcMinOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNhwcMinOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMinOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range PoolingNhwcMinOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNhwcMinOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNhwcMinOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNhwcMinOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void PoolingNhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNhwcMinOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNhwcMinOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNhwcMinOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult PoolingNhwcMinOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = (*this)->getAttr(stridesAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    auto tblgen_dilations = (*this)->getAttr(dilationsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
      return ::mlir::failure();
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

::mlir::LogicalResult PoolingNhwcMinOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcMinOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcMinUnsignedOp definitions
//===----------------------------------------------------------------------===//

PoolingNhwcMinUnsignedOpAdaptor::PoolingNhwcMinUnsignedOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

PoolingNhwcMinUnsignedOpAdaptor::PoolingNhwcMinUnsignedOpAdaptor(PoolingNhwcMinUnsignedOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange PoolingNhwcMinUnsignedOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNhwcMinUnsignedOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange PoolingNhwcMinUnsignedOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNhwcMinUnsignedOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNhwcMinUnsignedOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNhwcMinUnsignedOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::Region &PoolingNhwcMinUnsignedOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult PoolingNhwcMinUnsignedOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.pooling_nhwc_min_unsigned' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_nhwc_min_unsigned' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = odsAttrs.get("strides");
    if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.pooling_nhwc_min_unsigned' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
    auto tblgen_dilations = odsAttrs.get("dilations");
    if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.pooling_nhwc_min_unsigned' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNhwcMinUnsignedOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range PoolingNhwcMinUnsignedOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNhwcMinUnsignedOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNhwcMinUnsignedOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNhwcMinUnsignedOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMinUnsignedOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range PoolingNhwcMinUnsignedOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNhwcMinUnsignedOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMinUnsignedOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNhwcMinUnsignedOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNhwcMinUnsignedOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void PoolingNhwcMinUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNhwcMinUnsignedOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNhwcMinUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNhwcMinUnsignedOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNhwcMinUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNhwcMinUnsignedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNhwcMinUnsignedOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult PoolingNhwcMinUnsignedOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = (*this)->getAttr(stridesAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    auto tblgen_dilations = (*this)->getAttr(dilationsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
      return ::mlir::failure();
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

::mlir::LogicalResult PoolingNhwcMinUnsignedOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcMinUnsignedOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcSumOp definitions
//===----------------------------------------------------------------------===//

PoolingNhwcSumOpAdaptor::PoolingNhwcSumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

PoolingNhwcSumOpAdaptor::PoolingNhwcSumOpAdaptor(PoolingNhwcSumOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange PoolingNhwcSumOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNhwcSumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange PoolingNhwcSumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNhwcSumOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNhwcSumOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNhwcSumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::DenseIntElementsAttr PoolingNhwcSumOpAdaptor::dilationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
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

::mlir::Region &PoolingNhwcSumOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult PoolingNhwcSumOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.pooling_nhwc_sum' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.pooling_nhwc_sum' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = odsAttrs.get("strides");
    if (tblgen_strides && !((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.pooling_nhwc_sum' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
    auto tblgen_dilations = odsAttrs.get("dilations");
    if (tblgen_dilations && !((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2})))))
      return emitError(loc, "'linalg.pooling_nhwc_sum' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PoolingNhwcSumOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range PoolingNhwcSumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNhwcSumOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNhwcSumOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNhwcSumOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcSumOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range PoolingNhwcSumOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNhwcSumOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOp::strides() {
  auto attr = stridesAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOp::dilations() {
  auto attr = dilationsAttr();
    if (!attr)
      return ::mlir::DenseIntElementsAttr::get(::mlir::RankedTensorType::get({2}, ::mlir::Builder((*this)->getContext()).getIntegerType(64)), ::llvm::makeArrayRef({ static_cast<int64_t>(1), static_cast<int64_t>(1) }));
  return attr;
}

void PoolingNhwcSumOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNhwcSumOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void PoolingNhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNhwcSumOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNhwcSumOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNhwcSumOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
  
}

::mlir::LogicalResult PoolingNhwcSumOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_strides = (*this)->getAttr(stridesAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    auto tblgen_dilations = (*this)->getAttr(dilationsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgStructuredOps1(*this, tblgen_dilations, "dilations")))
      return ::mlir::failure();
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

::mlir::LogicalResult PoolingNhwcSumOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcSumOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::QuantizedBatchMatmulOp definitions
//===----------------------------------------------------------------------===//

QuantizedBatchMatmulOpAdaptor::QuantizedBatchMatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

QuantizedBatchMatmulOpAdaptor::QuantizedBatchMatmulOpAdaptor(QuantizedBatchMatmulOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange QuantizedBatchMatmulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> QuantizedBatchMatmulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange QuantizedBatchMatmulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange QuantizedBatchMatmulOpAdaptor::inputs() {
  return getODSOperands(0);
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

::mlir::Region &QuantizedBatchMatmulOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult QuantizedBatchMatmulOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.quantized_batch_matmul' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.quantized_batch_matmul' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> QuantizedBatchMatmulOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range QuantizedBatchMatmulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range QuantizedBatchMatmulOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range QuantizedBatchMatmulOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange QuantizedBatchMatmulOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange QuantizedBatchMatmulOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range QuantizedBatchMatmulOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &QuantizedBatchMatmulOp::region() {
  return (*this)->getRegion(0);
}

void QuantizedBatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<QuantizedBatchMatmulOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void QuantizedBatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<QuantizedBatchMatmulOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void QuantizedBatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult QuantizedBatchMatmulOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
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
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::QuantizedBatchMatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::QuantizedMatmulOp definitions
//===----------------------------------------------------------------------===//

QuantizedMatmulOpAdaptor::QuantizedMatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

QuantizedMatmulOpAdaptor::QuantizedMatmulOpAdaptor(QuantizedMatmulOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange QuantizedMatmulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> QuantizedMatmulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange QuantizedMatmulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange QuantizedMatmulOpAdaptor::inputs() {
  return getODSOperands(0);
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

::mlir::Region &QuantizedMatmulOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult QuantizedMatmulOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.quantized_matmul' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.quantized_matmul' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> QuantizedMatmulOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range QuantizedMatmulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range QuantizedMatmulOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range QuantizedMatmulOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange QuantizedMatmulOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange QuantizedMatmulOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range QuantizedMatmulOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &QuantizedMatmulOp::region() {
  return (*this)->getRegion(0);
}

void QuantizedMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<QuantizedMatmulOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void QuantizedMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<QuantizedMatmulOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void QuantizedMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult QuantizedMatmulOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
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
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::QuantizedMatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::SoftPlus2DOp definitions
//===----------------------------------------------------------------------===//

SoftPlus2DOpAdaptor::SoftPlus2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

SoftPlus2DOpAdaptor::SoftPlus2DOpAdaptor(SoftPlus2DOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange SoftPlus2DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SoftPlus2DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange SoftPlus2DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange SoftPlus2DOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange SoftPlus2DOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr SoftPlus2DOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange SoftPlus2DOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &SoftPlus2DOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult SoftPlus2DOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.soft_plus_2d' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.soft_plus_2d' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> SoftPlus2DOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range SoftPlus2DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range SoftPlus2DOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range SoftPlus2DOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange SoftPlus2DOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SoftPlus2DOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> SoftPlus2DOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range SoftPlus2DOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range SoftPlus2DOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &SoftPlus2DOp::region() {
  return (*this)->getRegion(0);
}

void SoftPlus2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<SoftPlus2DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void SoftPlus2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<SoftPlus2DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void SoftPlus2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult SoftPlus2DOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
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

::mlir::LogicalResult SoftPlus2DOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::SoftPlus2DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::VecmatOp definitions
//===----------------------------------------------------------------------===//

VecmatOpAdaptor::VecmatOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

VecmatOpAdaptor::VecmatOpAdaptor(VecmatOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange VecmatOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VecmatOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange VecmatOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange VecmatOpAdaptor::inputs() {
  return getODSOperands(0);
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

::mlir::Region &VecmatOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult VecmatOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'linalg.vecmat' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'linalg.vecmat' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> VecmatOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range VecmatOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range VecmatOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range VecmatOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange VecmatOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange VecmatOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::Operation::result_range VecmatOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &VecmatOp::region() {
  return (*this)->getRegion(0);
}

void VecmatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<VecmatOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void VecmatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<VecmatOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void VecmatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::LogicalResult VecmatOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
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
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::VecmatOp)


#endif  // GET_OP_CLASSES

