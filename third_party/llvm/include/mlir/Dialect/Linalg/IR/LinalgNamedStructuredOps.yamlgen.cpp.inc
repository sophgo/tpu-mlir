
//===----------------------------------------------------------------------===//
// Implementation of CopyOp
//===----------------------------------------------------------------------===//

ArrayAttr CopyOp::iterator_types() {
  int64_t rank = getRank(getOutputOperand(0));
  return Builder(getContext()).getStrArrayAttr(
    SmallVector<StringRef>(rank, getParallelIteratorTypeName()));
}

ArrayAttr CopyOp::indexing_maps() {
  MLIRContext *context = getContext();
  AffineMap scalarMap = AffineMap::get(getNumParallelLoops(), 0, context);
  AffineMap tensorMap = AffineMap::getMultiDimIdentityMap(
    getNumParallelLoops(), context);
  SmallVector<AffineMap> indexingMaps;
  for (OpOperand *opOperand : getInputAndOutputOperands())
    indexingMaps.push_back(getRank(opOperand) == 0 ? scalarMap : tensorMap);
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
}

unsigned CopyOp::getNumRegionArgs() { return 2; }

std::string CopyOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void CopyOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(2 > 0 && block.getNumArguments() == 2 &&
         "CopyOp regionBuilder expects 2 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
TypeFn castVal = TypeFn::cast_signed;
auto castIter = llvm::find_if(attrs, [&](const NamedAttribute &attr) {
                              return attr.getName() == "cast"; });
if (castIter != attrs.end()) {
  if (auto attr = castIter->getValue().dyn_cast<TypeFnAttr>())
    castVal = attr.getValue();
}

  Value value1 = helper.buildTypeFn(castVal, block.getArgument(1).getType(), block.getArgument(0));
  yields.push_back(value1);
  helper.yieldOutputs(yields);
}

ParseResult CopyOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<CopyOp>(parser, result);
}
void CopyOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult CopyOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void CopyOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of ElemwiseUnaryOp
//===----------------------------------------------------------------------===//

ArrayAttr ElemwiseUnaryOp::iterator_types() {
  int64_t rank = getRank(getOutputOperand(0));
  return Builder(getContext()).getStrArrayAttr(
    SmallVector<StringRef>(rank, getParallelIteratorTypeName()));
}

ArrayAttr ElemwiseUnaryOp::indexing_maps() {
  MLIRContext *context = getContext();
  AffineMap scalarMap = AffineMap::get(getNumParallelLoops(), 0, context);
  AffineMap tensorMap = AffineMap::getMultiDimIdentityMap(
    getNumParallelLoops(), context);
  SmallVector<AffineMap> indexingMaps;
  for (OpOperand *opOperand : getInputAndOutputOperands())
    indexingMaps.push_back(getRank(opOperand) == 0 ? scalarMap : tensorMap);
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
}

unsigned ElemwiseUnaryOp::getNumRegionArgs() { return 2; }

std::string ElemwiseUnaryOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void ElemwiseUnaryOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(2 > 0 && block.getNumArguments() == 2 &&
         "ElemwiseUnaryOp regionBuilder expects 2 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
UnaryFn funVal = UnaryFn::exp;
auto funIter = llvm::find_if(attrs, [&](const NamedAttribute &attr) {
                              return attr.getName() == "fun"; });
if (funIter != attrs.end()) {
  if (auto attr = funIter->getValue().dyn_cast<UnaryFnAttr>())
    funVal = attr.getValue();
}

  
TypeFn castVal = TypeFn::cast_signed;
auto castIter = llvm::find_if(attrs, [&](const NamedAttribute &attr) {
                              return attr.getName() == "cast"; });
if (castIter != attrs.end()) {
  if (auto attr = castIter->getValue().dyn_cast<TypeFnAttr>())
    castVal = attr.getValue();
}

  Value value1 = helper.buildTypeFn(castVal, block.getArgument(1).getType(), block.getArgument(0));
  Value value2 = helper.buildUnaryFn(funVal, value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult ElemwiseUnaryOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<ElemwiseUnaryOp>(parser, result);
}
void ElemwiseUnaryOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult ElemwiseUnaryOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void ElemwiseUnaryOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of ElemwiseBinaryOp
//===----------------------------------------------------------------------===//

ArrayAttr ElemwiseBinaryOp::iterator_types() {
  int64_t rank = getRank(getOutputOperand(0));
  return Builder(getContext()).getStrArrayAttr(
    SmallVector<StringRef>(rank, getParallelIteratorTypeName()));
}

ArrayAttr ElemwiseBinaryOp::indexing_maps() {
  MLIRContext *context = getContext();
  AffineMap scalarMap = AffineMap::get(getNumParallelLoops(), 0, context);
  AffineMap tensorMap = AffineMap::getMultiDimIdentityMap(
    getNumParallelLoops(), context);
  SmallVector<AffineMap> indexingMaps;
  for (OpOperand *opOperand : getInputAndOutputOperands())
    indexingMaps.push_back(getRank(opOperand) == 0 ? scalarMap : tensorMap);
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
}

unsigned ElemwiseBinaryOp::getNumRegionArgs() { return 3; }

std::string ElemwiseBinaryOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void ElemwiseBinaryOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "ElemwiseBinaryOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
BinaryFn funVal = BinaryFn::add;
auto funIter = llvm::find_if(attrs, [&](const NamedAttribute &attr) {
                              return attr.getName() == "fun"; });
if (funIter != attrs.end()) {
  if (auto attr = funIter->getValue().dyn_cast<BinaryFnAttr>())
    funVal = attr.getValue();
}

  
TypeFn castVal = TypeFn::cast_signed;
auto castIter = llvm::find_if(attrs, [&](const NamedAttribute &attr) {
                              return attr.getName() == "cast"; });
if (castIter != attrs.end()) {
  if (auto attr = castIter->getValue().dyn_cast<TypeFnAttr>())
    castVal = attr.getValue();
}

  Value value1 = helper.buildTypeFn(castVal, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(castVal, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(funVal, value1, value2);
  yields.push_back(value3);
  helper.yieldOutputs(yields);
}

ParseResult ElemwiseBinaryOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<ElemwiseBinaryOp>(parser, result);
}
void ElemwiseBinaryOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult ElemwiseBinaryOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void ElemwiseBinaryOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of MatmulOp
//===----------------------------------------------------------------------===//

ArrayAttr MatmulOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(MatmulOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  return exprs;
}

ArrayAttr MatmulOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned MatmulOp::getNumRegionArgs() { return 3; }

std::string MatmulOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void MatmulOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "MatmulOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
TypeFn castVal = TypeFn::cast_signed;
auto castIter = llvm::find_if(attrs, [&](const NamedAttribute &attr) {
                              return attr.getName() == "cast"; });
if (castIter != attrs.end()) {
  if (auto attr = castIter->getValue().dyn_cast<TypeFnAttr>())
    castVal = attr.getValue();
}

  Value value1 = helper.buildTypeFn(castVal, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(castVal, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult MatmulOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<MatmulOp>(parser, result);
}
void MatmulOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult MatmulOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void MatmulOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of MatmulUnsignedOp
//===----------------------------------------------------------------------===//

ArrayAttr MatmulUnsignedOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(MatmulUnsignedOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  return exprs;
}

ArrayAttr MatmulUnsignedOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned MatmulUnsignedOp::getNumRegionArgs() { return 3; }

std::string MatmulUnsignedOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void MatmulUnsignedOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "MatmulUnsignedOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_unsigned, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_unsigned, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult MatmulUnsignedOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<MatmulUnsignedOp>(parser, result);
}
void MatmulUnsignedOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult MatmulUnsignedOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void MatmulUnsignedOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of QuantizedMatmulOp
//===----------------------------------------------------------------------===//

ArrayAttr QuantizedMatmulOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(QuantizedMatmulOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  return exprs;
}

ArrayAttr QuantizedMatmulOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned QuantizedMatmulOp::getNumRegionArgs() { return 5; }

std::string QuantizedMatmulOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void QuantizedMatmulOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(5 > 0 && block.getNumArguments() == 5 &&
         "QuantizedMatmulOp regionBuilder expects 5 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(2));
  Value value3 = helper.buildBinaryFn(BinaryFn::sub, value1, value2);
  Value value4 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(1));
  Value value5 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(3));
  Value value6 = helper.buildBinaryFn(BinaryFn::sub, value4, value5);
  Value value7 = helper.buildBinaryFn(BinaryFn::mul, value3, value6);
  Value value8 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(4), value7);
  yields.push_back(value8);
  helper.yieldOutputs(yields);
}

ParseResult QuantizedMatmulOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<QuantizedMatmulOp>(parser, result);
}
void QuantizedMatmulOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult QuantizedMatmulOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void QuantizedMatmulOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of Mmt4DOp
//===----------------------------------------------------------------------===//

ArrayAttr Mmt4DOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(Mmt4DOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));
  exprs.push_back(getAffineSymbolExpr(4, context));
  exprs.push_back(getAffineSymbolExpr(5, context));
  return exprs;
}

ArrayAttr Mmt4DOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5] -> (d0, d2, d3, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5] -> (d1, d2, d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5] -> (d0, d1, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Mmt4DOp::getNumRegionArgs() { return 3; }

std::string Mmt4DOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void Mmt4DOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Mmt4DOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Mmt4DOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<Mmt4DOp>(parser, result);
}
void Mmt4DOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult Mmt4DOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void Mmt4DOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of BatchMatmulOp
//===----------------------------------------------------------------------===//

ArrayAttr BatchMatmulOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(BatchMatmulOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));
  return exprs;
}

ArrayAttr BatchMatmulOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d1, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d3, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned BatchMatmulOp::getNumRegionArgs() { return 3; }

std::string BatchMatmulOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void BatchMatmulOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "BatchMatmulOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult BatchMatmulOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<BatchMatmulOp>(parser, result);
}
void BatchMatmulOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult BatchMatmulOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void BatchMatmulOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of QuantizedBatchMatmulOp
//===----------------------------------------------------------------------===//

ArrayAttr QuantizedBatchMatmulOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(QuantizedBatchMatmulOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));
  return exprs;
}

ArrayAttr QuantizedBatchMatmulOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d1, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d3, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned QuantizedBatchMatmulOp::getNumRegionArgs() { return 5; }

std::string QuantizedBatchMatmulOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void QuantizedBatchMatmulOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(5 > 0 && block.getNumArguments() == 5 &&
         "QuantizedBatchMatmulOp regionBuilder expects 5 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(2));
  Value value3 = helper.buildBinaryFn(BinaryFn::sub, value1, value2);
  Value value4 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(1));
  Value value5 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(3));
  Value value6 = helper.buildBinaryFn(BinaryFn::sub, value4, value5);
  Value value7 = helper.buildBinaryFn(BinaryFn::mul, value3, value6);
  Value value8 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(4), value7);
  yields.push_back(value8);
  helper.yieldOutputs(yields);
}

ParseResult QuantizedBatchMatmulOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<QuantizedBatchMatmulOp>(parser, result);
}
void QuantizedBatchMatmulOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult QuantizedBatchMatmulOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void QuantizedBatchMatmulOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of MatvecOp
//===----------------------------------------------------------------------===//

ArrayAttr MatvecOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(MatvecOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  return exprs;
}

ArrayAttr MatvecOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned MatvecOp::getNumRegionArgs() { return 3; }

std::string MatvecOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void MatvecOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "MatvecOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult MatvecOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<MatvecOp>(parser, result);
}
void MatvecOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult MatvecOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void MatvecOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of VecmatOp
//===----------------------------------------------------------------------===//

ArrayAttr VecmatOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(VecmatOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  return exprs;
}

ArrayAttr VecmatOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d1, d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned VecmatOp::getNumRegionArgs() { return 3; }

std::string VecmatOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void VecmatOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "VecmatOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult VecmatOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<VecmatOp>(parser, result);
}
void VecmatOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult VecmatOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void VecmatOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of BatchMatvecOp
//===----------------------------------------------------------------------===//

ArrayAttr BatchMatvecOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(BatchMatvecOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  return exprs;
}

ArrayAttr BatchMatvecOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned BatchMatvecOp::getNumRegionArgs() { return 3; }

std::string BatchMatvecOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void BatchMatvecOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "BatchMatvecOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult BatchMatvecOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<BatchMatvecOp>(parser, result);
}
void BatchMatvecOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult BatchMatvecOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void BatchMatvecOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of DotOp
//===----------------------------------------------------------------------===//

ArrayAttr DotOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(DotOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  return exprs;
}

ArrayAttr DotOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0)[s0] -> (d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 1, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0)[s0] -> (d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 1, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0)[s0] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 1, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DotOp::getNumRegionArgs() { return 3; }

std::string DotOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void DotOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "DotOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult DotOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<DotOp>(parser, result);
}
void DotOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult DotOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void DotOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of Conv1DOp
//===----------------------------------------------------------------------===//

ArrayAttr Conv1DOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(Conv1DOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  return exprs;
}

ArrayAttr Conv1DOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0 + d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv1DOp::getNumRegionArgs() { return 3; }

std::string Conv1DOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void Conv1DOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv1DOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Conv1DOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<Conv1DOp>(parser, result);
}
void Conv1DOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult Conv1DOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void Conv1DOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of Conv2DOp
//===----------------------------------------------------------------------===//

ArrayAttr Conv2DOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(Conv2DOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));
  return exprs;
}

ArrayAttr Conv2DOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 + d2, d1 + d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv2DOp::getNumRegionArgs() { return 3; }

std::string Conv2DOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void Conv2DOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv2DOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Conv2DOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<Conv2DOp>(parser, result);
}
void Conv2DOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult Conv2DOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void Conv2DOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of Conv3DOp
//===----------------------------------------------------------------------===//

ArrayAttr Conv3DOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(Conv3DOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));
  exprs.push_back(getAffineSymbolExpr(4, context));
  exprs.push_back(getAffineSymbolExpr(5, context));
  return exprs;
}

ArrayAttr Conv3DOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5] -> (d0 + d3, d1 + d4, d2 + d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5] -> (d3, d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv3DOp::getNumRegionArgs() { return 3; }

std::string Conv3DOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void Conv3DOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv3DOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Conv3DOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<Conv3DOp>(parser, result);
}
void Conv3DOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult Conv3DOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void Conv3DOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of Conv1DNwcWcfOp
//===----------------------------------------------------------------------===//

ArrayAttr Conv1DNwcWcfOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(Conv1DNwcWcfOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));
  exprs.push_back(getAffineSymbolExpr(6, context));
  return exprs;
}

ArrayAttr Conv1DNwcWcfOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4)[s0, s1, s2, s3, s4, s5, s6] -> (d0, d1 * s2 + d3 * s4, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 5, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4)[s0, s1, s2, s3, s4, s5, s6] -> (d3, d4, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 5, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4)[s0, s1, s2, s3, s4, s5, s6] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 5, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv1DNwcWcfOp::getNumRegionArgs() { return 3; }

std::string Conv1DNwcWcfOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool Conv1DNwcWcfOp::hasDynamicIndexingMaps() { return true; }
LogicalResult Conv1DNwcWcfOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void Conv1DNwcWcfOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv1DNwcWcfOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Conv1DNwcWcfOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<Conv1DNwcWcfOp>(parser, result);
}
void Conv1DNwcWcfOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult Conv1DNwcWcfOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void Conv1DNwcWcfOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of Conv2DNhwcHwcfOp
//===----------------------------------------------------------------------===//

ArrayAttr Conv2DNhwcHwcfOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(Conv2DNhwcHwcfOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  exprs.push_back(getAffineSymbolExpr(10, context));
  return exprs;
}

ArrayAttr Conv2DNhwcHwcfOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d6)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d4, d5, d6, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv2DNhwcHwcfOp::getNumRegionArgs() { return 3; }

std::string Conv2DNhwcHwcfOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool Conv2DNhwcHwcfOp::hasDynamicIndexingMaps() { return true; }
LogicalResult Conv2DNhwcHwcfOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void Conv2DNhwcHwcfOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv2DNhwcHwcfOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Conv2DNhwcHwcfOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<Conv2DNhwcHwcfOp>(parser, result);
}
void Conv2DNhwcHwcfOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult Conv2DNhwcHwcfOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void Conv2DNhwcHwcfOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of Conv2DNhwcHwcfQOp
//===----------------------------------------------------------------------===//

ArrayAttr Conv2DNhwcHwcfQOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(Conv2DNhwcHwcfQOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  exprs.push_back(getAffineSymbolExpr(10, context));
  return exprs;
}

ArrayAttr Conv2DNhwcHwcfQOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d6)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d4, d5, d6, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv2DNhwcHwcfQOp::getNumRegionArgs() { return 5; }

std::string Conv2DNhwcHwcfQOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool Conv2DNhwcHwcfQOp::hasDynamicIndexingMaps() { return true; }
LogicalResult Conv2DNhwcHwcfQOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void Conv2DNhwcHwcfQOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(5 > 0 && block.getNumArguments() == 5 &&
         "Conv2DNhwcHwcfQOp regionBuilder expects 5 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(2));
  Value value3 = helper.buildBinaryFn(BinaryFn::sub, value1, value2);
  Value value4 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(1));
  Value value5 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(3));
  Value value6 = helper.buildBinaryFn(BinaryFn::sub, value4, value5);
  Value value7 = helper.buildBinaryFn(BinaryFn::mul, value3, value6);
  Value value8 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(4), value7);
  yields.push_back(value8);
  helper.yieldOutputs(yields);
}

ParseResult Conv2DNhwcHwcfQOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<Conv2DNhwcHwcfQOp>(parser, result);
}
void Conv2DNhwcHwcfQOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult Conv2DNhwcHwcfQOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void Conv2DNhwcHwcfQOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of Conv2DNchwFchwOp
//===----------------------------------------------------------------------===//

ArrayAttr Conv2DNchwFchwOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(Conv2DNchwFchwOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));

int64_t cst3 = self.strides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst3, context));

  exprs.push_back(getAffineSymbolExpr(4, context));

int64_t cst5 = self.dilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst5, context));

  exprs.push_back(getAffineSymbolExpr(6, context));

int64_t cst7 = self.strides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst7, context));

  exprs.push_back(getAffineSymbolExpr(8, context));

int64_t cst9 = self.dilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst9, context));

  exprs.push_back(getAffineSymbolExpr(10, context));
  return exprs;
}

ArrayAttr Conv2DNchwFchwOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d4, d2 * s3 + d5 * s5, d3 * s7 + d6 * s9)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d1, d4, d5, d6)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv2DNchwFchwOp::getNumRegionArgs() { return 3; }

std::string Conv2DNchwFchwOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool Conv2DNchwFchwOp::hasDynamicIndexingMaps() { return true; }
LogicalResult Conv2DNchwFchwOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void Conv2DNchwFchwOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv2DNchwFchwOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Conv2DNchwFchwOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<Conv2DNchwFchwOp>(parser, result);
}
void Conv2DNchwFchwOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult Conv2DNchwFchwOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void Conv2DNchwFchwOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of Conv3DNdhwcDhwcfOp
//===----------------------------------------------------------------------===//

ArrayAttr Conv3DNdhwcDhwcfOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(Conv3DNdhwcDhwcfOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));

int64_t cst10 = self.strides().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst10, context));

  exprs.push_back(getAffineSymbolExpr(11, context));

int64_t cst12 = self.dilations().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst12, context));

  exprs.push_back(getAffineSymbolExpr(13, context));
  exprs.push_back(getAffineSymbolExpr(14, context));
  return exprs;
}

ArrayAttr Conv3DNdhwcDhwcfOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14] -> (d0, d1 * s2 + d5 * s4, d2 * s6 + d6 * s8, d3 * s10 + d7 * s12, d8)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 9, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14] -> (d5, d6, d7, d8, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 9, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14] -> (d0, d1, d2, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 9, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv3DNdhwcDhwcfOp::getNumRegionArgs() { return 3; }

std::string Conv3DNdhwcDhwcfOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool Conv3DNdhwcDhwcfOp::hasDynamicIndexingMaps() { return true; }
LogicalResult Conv3DNdhwcDhwcfOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void Conv3DNdhwcDhwcfOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv3DNdhwcDhwcfOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Conv3DNdhwcDhwcfOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<Conv3DNdhwcDhwcfOp>(parser, result);
}
void Conv3DNdhwcDhwcfOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult Conv3DNdhwcDhwcfOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void Conv3DNdhwcDhwcfOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of DepthwiseConv1DNwcWcOp
//===----------------------------------------------------------------------===//

ArrayAttr DepthwiseConv1DNwcWcOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(DepthwiseConv1DNwcWcOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));
  return exprs;
}

ArrayAttr DepthwiseConv1DNwcWcOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d0, d1 * s2 + d3 * s4, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d3, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DepthwiseConv1DNwcWcOp::getNumRegionArgs() { return 3; }

std::string DepthwiseConv1DNwcWcOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool DepthwiseConv1DNwcWcOp::hasDynamicIndexingMaps() { return true; }
LogicalResult DepthwiseConv1DNwcWcOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void DepthwiseConv1DNwcWcOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "DepthwiseConv1DNwcWcOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult DepthwiseConv1DNwcWcOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<DepthwiseConv1DNwcWcOp>(parser, result);
}
void DepthwiseConv1DNwcWcOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult DepthwiseConv1DNwcWcOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void DepthwiseConv1DNwcWcOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of DepthwiseConv2DNhwcHwcOp
//===----------------------------------------------------------------------===//

ArrayAttr DepthwiseConv2DNhwcHwcOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(DepthwiseConv2DNhwcHwcOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  return exprs;
}

ArrayAttr DepthwiseConv2DNhwcHwcOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d4, d5, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DepthwiseConv2DNhwcHwcOp::getNumRegionArgs() { return 3; }

std::string DepthwiseConv2DNhwcHwcOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool DepthwiseConv2DNhwcHwcOp::hasDynamicIndexingMaps() { return true; }
LogicalResult DepthwiseConv2DNhwcHwcOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void DepthwiseConv2DNhwcHwcOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "DepthwiseConv2DNhwcHwcOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult DepthwiseConv2DNhwcHwcOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<DepthwiseConv2DNhwcHwcOp>(parser, result);
}
void DepthwiseConv2DNhwcHwcOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult DepthwiseConv2DNhwcHwcOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void DepthwiseConv2DNhwcHwcOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of DepthwiseConv2DNhwcHwcQOp
//===----------------------------------------------------------------------===//

ArrayAttr DepthwiseConv2DNhwcHwcQOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(DepthwiseConv2DNhwcHwcQOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  return exprs;
}

ArrayAttr DepthwiseConv2DNhwcHwcQOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d4, d5, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DepthwiseConv2DNhwcHwcQOp::getNumRegionArgs() { return 5; }

std::string DepthwiseConv2DNhwcHwcQOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool DepthwiseConv2DNhwcHwcQOp::hasDynamicIndexingMaps() { return true; }
LogicalResult DepthwiseConv2DNhwcHwcQOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void DepthwiseConv2DNhwcHwcQOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(5 > 0 && block.getNumArguments() == 5 &&
         "DepthwiseConv2DNhwcHwcQOp regionBuilder expects 5 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(2));
  Value value3 = helper.buildBinaryFn(BinaryFn::sub, value1, value2);
  Value value4 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(1));
  Value value5 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(3));
  Value value6 = helper.buildBinaryFn(BinaryFn::sub, value4, value5);
  Value value7 = helper.buildBinaryFn(BinaryFn::mul, value3, value6);
  Value value8 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(4), value7);
  yields.push_back(value8);
  helper.yieldOutputs(yields);
}

ParseResult DepthwiseConv2DNhwcHwcQOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<DepthwiseConv2DNhwcHwcQOp>(parser, result);
}
void DepthwiseConv2DNhwcHwcQOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult DepthwiseConv2DNhwcHwcQOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void DepthwiseConv2DNhwcHwcQOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of DepthwiseConv2DNhwcHwcmOp
//===----------------------------------------------------------------------===//

ArrayAttr DepthwiseConv2DNhwcHwcmOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(DepthwiseConv2DNhwcHwcmOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  exprs.push_back(getAffineSymbolExpr(10, context));
  return exprs;
}

ArrayAttr DepthwiseConv2DNhwcHwcmOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1 * s2 + d5 * s4, d2 * s6 + d6 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d5, d6, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1, d2, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DepthwiseConv2DNhwcHwcmOp::getNumRegionArgs() { return 3; }

std::string DepthwiseConv2DNhwcHwcmOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool DepthwiseConv2DNhwcHwcmOp::hasDynamicIndexingMaps() { return true; }
LogicalResult DepthwiseConv2DNhwcHwcmOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void DepthwiseConv2DNhwcHwcmOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "DepthwiseConv2DNhwcHwcmOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult DepthwiseConv2DNhwcHwcmOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<DepthwiseConv2DNhwcHwcmOp>(parser, result);
}
void DepthwiseConv2DNhwcHwcmOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult DepthwiseConv2DNhwcHwcmOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void DepthwiseConv2DNhwcHwcmOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of DepthwiseConv2DNhwcHwcmQOp
//===----------------------------------------------------------------------===//

ArrayAttr DepthwiseConv2DNhwcHwcmQOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(DepthwiseConv2DNhwcHwcmQOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  exprs.push_back(getAffineSymbolExpr(10, context));
  return exprs;
}

ArrayAttr DepthwiseConv2DNhwcHwcmQOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1 * s2 + d5 * s4, d2 * s6 + d6 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d5, d6, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1, d2, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DepthwiseConv2DNhwcHwcmQOp::getNumRegionArgs() { return 5; }

std::string DepthwiseConv2DNhwcHwcmQOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool DepthwiseConv2DNhwcHwcmQOp::hasDynamicIndexingMaps() { return true; }
LogicalResult DepthwiseConv2DNhwcHwcmQOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void DepthwiseConv2DNhwcHwcmQOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(5 > 0 && block.getNumArguments() == 5 &&
         "DepthwiseConv2DNhwcHwcmQOp regionBuilder expects 5 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(2));
  Value value3 = helper.buildBinaryFn(BinaryFn::sub, value1, value2);
  Value value4 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(1));
  Value value5 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(3));
  Value value6 = helper.buildBinaryFn(BinaryFn::sub, value4, value5);
  Value value7 = helper.buildBinaryFn(BinaryFn::mul, value3, value6);
  Value value8 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(4), value7);
  yields.push_back(value8);
  helper.yieldOutputs(yields);
}

ParseResult DepthwiseConv2DNhwcHwcmQOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<DepthwiseConv2DNhwcHwcmQOp>(parser, result);
}
void DepthwiseConv2DNhwcHwcmQOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult DepthwiseConv2DNhwcHwcmQOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void DepthwiseConv2DNhwcHwcmQOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNhwcSumOp
//===----------------------------------------------------------------------===//

ArrayAttr PoolingNhwcSumOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNhwcSumOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  return exprs;
}

ArrayAttr PoolingNhwcSumOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNhwcSumOp::getNumRegionArgs() { return 3; }

std::string PoolingNhwcSumOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNhwcSumOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNhwcSumOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNhwcSumOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNhwcSumOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNhwcSumOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<PoolingNhwcSumOp>(parser, result);
}
void PoolingNhwcSumOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult PoolingNhwcSumOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void PoolingNhwcSumOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNhwcMaxOp
//===----------------------------------------------------------------------===//

ArrayAttr PoolingNhwcMaxOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNhwcMaxOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  return exprs;
}

ArrayAttr PoolingNhwcMaxOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNhwcMaxOp::getNumRegionArgs() { return 3; }

std::string PoolingNhwcMaxOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNhwcMaxOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNhwcMaxOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNhwcMaxOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNhwcMaxOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::max_signed, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNhwcMaxOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<PoolingNhwcMaxOp>(parser, result);
}
void PoolingNhwcMaxOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult PoolingNhwcMaxOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void PoolingNhwcMaxOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNhwcMaxUnsignedOp
//===----------------------------------------------------------------------===//

ArrayAttr PoolingNhwcMaxUnsignedOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNhwcMaxUnsignedOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  return exprs;
}

ArrayAttr PoolingNhwcMaxUnsignedOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNhwcMaxUnsignedOp::getNumRegionArgs() { return 3; }

std::string PoolingNhwcMaxUnsignedOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNhwcMaxUnsignedOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNhwcMaxUnsignedOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNhwcMaxUnsignedOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNhwcMaxUnsignedOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_unsigned, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::max_unsigned, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNhwcMaxUnsignedOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<PoolingNhwcMaxUnsignedOp>(parser, result);
}
void PoolingNhwcMaxUnsignedOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult PoolingNhwcMaxUnsignedOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void PoolingNhwcMaxUnsignedOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNchwMaxOp
//===----------------------------------------------------------------------===//

ArrayAttr PoolingNchwMaxOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNchwMaxOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));

int64_t cst3 = self.strides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst3, context));

  exprs.push_back(getAffineSymbolExpr(4, context));

int64_t cst5 = self.dilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst5, context));

  exprs.push_back(getAffineSymbolExpr(6, context));

int64_t cst7 = self.strides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst7, context));

  exprs.push_back(getAffineSymbolExpr(8, context));

int64_t cst9 = self.dilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst9, context));

  return exprs;
}

ArrayAttr PoolingNchwMaxOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2 * s3 + d4 * s5, d3 * s7 + d5 * s9)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNchwMaxOp::getNumRegionArgs() { return 3; }

std::string PoolingNchwMaxOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNchwMaxOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNchwMaxOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNchwMaxOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNchwMaxOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::max_signed, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNchwMaxOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<PoolingNchwMaxOp>(parser, result);
}
void PoolingNchwMaxOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult PoolingNchwMaxOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void PoolingNchwMaxOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNhwcMinOp
//===----------------------------------------------------------------------===//

ArrayAttr PoolingNhwcMinOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNhwcMinOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  return exprs;
}

ArrayAttr PoolingNhwcMinOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNhwcMinOp::getNumRegionArgs() { return 3; }

std::string PoolingNhwcMinOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNhwcMinOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNhwcMinOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNhwcMinOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNhwcMinOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::min_signed, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNhwcMinOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<PoolingNhwcMinOp>(parser, result);
}
void PoolingNhwcMinOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult PoolingNhwcMinOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void PoolingNhwcMinOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNhwcMinUnsignedOp
//===----------------------------------------------------------------------===//

ArrayAttr PoolingNhwcMinUnsignedOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNhwcMinUnsignedOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  return exprs;
}

ArrayAttr PoolingNhwcMinUnsignedOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNhwcMinUnsignedOp::getNumRegionArgs() { return 3; }

std::string PoolingNhwcMinUnsignedOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNhwcMinUnsignedOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNhwcMinUnsignedOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNhwcMinUnsignedOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNhwcMinUnsignedOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_unsigned, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::min_unsigned, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNhwcMinUnsignedOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<PoolingNhwcMinUnsignedOp>(parser, result);
}
void PoolingNhwcMinUnsignedOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult PoolingNhwcMinUnsignedOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void PoolingNhwcMinUnsignedOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNdhwcSumOp
//===----------------------------------------------------------------------===//

ArrayAttr PoolingNdhwcSumOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNdhwcSumOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));

int64_t cst10 = self.strides().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst10, context));

  exprs.push_back(getAffineSymbolExpr(11, context));

int64_t cst12 = self.dilations().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst12, context));

  exprs.push_back(getAffineSymbolExpr(13, context));
  return exprs;
}

ArrayAttr PoolingNdhwcSumOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d0, d1 * s2 + d5 * s4, d2 * s6 + d6 * s8, d3 * s10 + d7 * s12, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d5, d6, d7)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d0, d1, d2, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNdhwcSumOp::getNumRegionArgs() { return 3; }

std::string PoolingNdhwcSumOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNdhwcSumOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNdhwcSumOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNdhwcSumOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNdhwcSumOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNdhwcSumOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<PoolingNdhwcSumOp>(parser, result);
}
void PoolingNdhwcSumOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult PoolingNdhwcSumOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void PoolingNdhwcSumOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNdhwcMaxOp
//===----------------------------------------------------------------------===//

ArrayAttr PoolingNdhwcMaxOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNdhwcMaxOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));

int64_t cst10 = self.strides().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst10, context));

  exprs.push_back(getAffineSymbolExpr(11, context));

int64_t cst12 = self.dilations().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst12, context));

  exprs.push_back(getAffineSymbolExpr(13, context));
  return exprs;
}

ArrayAttr PoolingNdhwcMaxOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d0, d1 * s2 + d5 * s4, d2 * s6 + d6 * s8, d3 * s10 + d7 * s12, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d5, d6, d7)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d0, d1, d2, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNdhwcMaxOp::getNumRegionArgs() { return 3; }

std::string PoolingNdhwcMaxOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNdhwcMaxOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNdhwcMaxOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNdhwcMaxOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNdhwcMaxOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::max_signed, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNdhwcMaxOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<PoolingNdhwcMaxOp>(parser, result);
}
void PoolingNdhwcMaxOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult PoolingNdhwcMaxOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void PoolingNdhwcMaxOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNdhwcMinOp
//===----------------------------------------------------------------------===//

ArrayAttr PoolingNdhwcMinOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNdhwcMinOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));

int64_t cst10 = self.strides().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst10, context));

  exprs.push_back(getAffineSymbolExpr(11, context));

int64_t cst12 = self.dilations().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst12, context));

  exprs.push_back(getAffineSymbolExpr(13, context));
  return exprs;
}

ArrayAttr PoolingNdhwcMinOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d0, d1 * s2 + d5 * s4, d2 * s6 + d6 * s8, d3 * s10 + d7 * s12, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d5, d6, d7)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d0, d1, d2, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNdhwcMinOp::getNumRegionArgs() { return 3; }

std::string PoolingNdhwcMinOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNdhwcMinOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNdhwcMinOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNdhwcMinOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNdhwcMinOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::min_signed, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNdhwcMinOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<PoolingNdhwcMinOp>(parser, result);
}
void PoolingNdhwcMinOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult PoolingNdhwcMinOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void PoolingNdhwcMinOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of FillOp
//===----------------------------------------------------------------------===//

ArrayAttr FillOp::iterator_types() {
  int64_t rank = getRank(getOutputOperand(0));
  return Builder(getContext()).getStrArrayAttr(
    SmallVector<StringRef>(rank, getParallelIteratorTypeName()));
}

ArrayAttr FillOp::indexing_maps() {
  MLIRContext *context = getContext();
  AffineMap scalarMap = AffineMap::get(getNumParallelLoops(), 0, context);
  AffineMap tensorMap = AffineMap::getMultiDimIdentityMap(
    getNumParallelLoops(), context);
  SmallVector<AffineMap> indexingMaps;
  for (OpOperand *opOperand : getInputAndOutputOperands())
    indexingMaps.push_back(getRank(opOperand) == 0 ? scalarMap : tensorMap);
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
}

unsigned FillOp::getNumRegionArgs() { return 2; }

std::string FillOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void FillOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(2 > 0 && block.getNumArguments() == 2 &&
         "FillOp regionBuilder expects 2 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(1).getType(), block.getArgument(0));
  yields.push_back(value1);
  helper.yieldOutputs(yields);
}

ParseResult FillOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<FillOp>(parser, result);
}
void FillOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult FillOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void FillOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of FillRng2DOp
//===----------------------------------------------------------------------===//

ArrayAttr FillRng2DOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(FillRng2DOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  return exprs;
}

ArrayAttr FillRng2DOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned FillRng2DOp::getNumRegionArgs() { return 4; }

std::string FillRng2DOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void FillRng2DOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(4 > 0 && block.getNumArguments() == 4 &&
         "FillRng2DOp regionBuilder expects 4 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.constant("2147483647 : i64");
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, helper.getFloat64Type(), value1);
  Value value3 = helper.index(1);
  Value value4 = helper.buildTypeFn(TypeFn::cast_signed, helper.getIntegerType(32), value3);
  Value value5 = helper.index(0);
  Value value6 = helper.buildTypeFn(TypeFn::cast_signed, helper.getIntegerType(32), value5);
  Value value7 = helper.buildBinaryFn(BinaryFn::add, value6, block.getArgument(2));
  Value value8 = helper.constant("1103515245 : i64");
  Value value9 = helper.buildTypeFn(TypeFn::cast_signed, helper.getIntegerType(32), value8);
  Value value10 = helper.buildBinaryFn(BinaryFn::mul, value7, value9);
  Value value11 = helper.constant("12345 : i64");
  Value value12 = helper.buildTypeFn(TypeFn::cast_signed, helper.getIntegerType(32), value11);
  Value value13 = helper.buildBinaryFn(BinaryFn::add, value10, value12);
  Value value14 = helper.buildBinaryFn(BinaryFn::add, value4, value13);
  Value value15 = helper.constant("1103515245 : i64");
  Value value16 = helper.buildTypeFn(TypeFn::cast_signed, helper.getIntegerType(32), value15);
  Value value17 = helper.buildBinaryFn(BinaryFn::mul, value14, value16);
  Value value18 = helper.constant("12345 : i64");
  Value value19 = helper.buildTypeFn(TypeFn::cast_signed, helper.getIntegerType(32), value18);
  Value value20 = helper.buildBinaryFn(BinaryFn::add, value17, value19);
  Value value21 = helper.buildTypeFn(TypeFn::cast_signed, helper.getFloat64Type(), value20);
  Value value22 = helper.buildBinaryFn(BinaryFn::add, value2, value21);
  Value value23 = helper.buildBinaryFn(BinaryFn::sub, block.getArgument(1), block.getArgument(0));
  Value value24 = helper.constant("2.3283063999999999E-10 : f64");
  Value value25 = helper.buildTypeFn(TypeFn::cast_signed, helper.getFloat64Type(), value24);
  Value value26 = helper.buildBinaryFn(BinaryFn::mul, value23, value25);
  Value value27 = helper.buildBinaryFn(BinaryFn::mul, value22, value26);
  Value value28 = helper.buildBinaryFn(BinaryFn::add, value27, block.getArgument(0));
  Value value29 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(3).getType(), value28);
  yields.push_back(value29);
  helper.yieldOutputs(yields);
}

ParseResult FillRng2DOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp<FillRng2DOp>(parser, result);
}
void FillRng2DOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, *this);
}

LogicalResult FillRng2DOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void FillRng2DOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}
