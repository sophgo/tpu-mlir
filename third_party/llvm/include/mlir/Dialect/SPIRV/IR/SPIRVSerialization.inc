/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* SPIR-V Serialization Utilities/Functions                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_SERIALIZATION_FNS

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AccessChainOp>(::mlir::spirv::AccessChainOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAccessChain, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicAndOp>(::mlir::spirv::AtomicAndOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemorySemanticsAttr>().getValue()))));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicAnd, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicCompareExchangeOp>(::mlir::spirv::AtomicCompareExchangeOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("equal_semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemorySemanticsAttr>().getValue()))));
    }
    elidedAttrs.push_back("equal_semantics");
  }
  {
    if (auto attr = op->getAttr("unequal_semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemorySemanticsAttr>().getValue()))));
    }
    elidedAttrs.push_back("unequal_semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(2)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #2 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicCompareExchange, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicCompareExchangeWeakOp>(::mlir::spirv::AtomicCompareExchangeWeakOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("equal_semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemorySemanticsAttr>().getValue()))));
    }
    elidedAttrs.push_back("equal_semantics");
  }
  {
    if (auto attr = op->getAttr("unequal_semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemorySemanticsAttr>().getValue()))));
    }
    elidedAttrs.push_back("unequal_semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(2)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #2 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicCompareExchangeWeak, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicExchangeOp>(::mlir::spirv::AtomicExchangeOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemorySemanticsAttr>().getValue()))));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicExchange, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicIAddOp>(::mlir::spirv::AtomicIAddOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemorySemanticsAttr>().getValue()))));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicIAdd, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicIDecrementOp>(::mlir::spirv::AtomicIDecrementOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op->getAttr("memory_scope")) {
    operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
  }
  elidedAttrs.push_back("memory_scope");
  if (auto attr = op->getAttr("semantics")) {
    operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemorySemanticsAttr>().getValue()))));
  }
  elidedAttrs.push_back("semantics");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicIDecrement, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicIIncrementOp>(::mlir::spirv::AtomicIIncrementOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op->getAttr("memory_scope")) {
    operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
  }
  elidedAttrs.push_back("memory_scope");
  if (auto attr = op->getAttr("semantics")) {
    operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemorySemanticsAttr>().getValue()))));
  }
  elidedAttrs.push_back("semantics");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicIIncrement, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicISubOp>(::mlir::spirv::AtomicISubOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemorySemanticsAttr>().getValue()))));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicISub, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicOrOp>(::mlir::spirv::AtomicOrOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemorySemanticsAttr>().getValue()))));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicOr, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicSMaxOp>(::mlir::spirv::AtomicSMaxOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemorySemanticsAttr>().getValue()))));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicSMax, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicSMinOp>(::mlir::spirv::AtomicSMinOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemorySemanticsAttr>().getValue()))));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicSMin, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicUMaxOp>(::mlir::spirv::AtomicUMaxOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemorySemanticsAttr>().getValue()))));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicUMax, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicUMinOp>(::mlir::spirv::AtomicUMinOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemorySemanticsAttr>().getValue()))));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicUMin, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicXorOp>(::mlir::spirv::AtomicXorOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemorySemanticsAttr>().getValue()))));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicXor, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitCountOp>(::mlir::spirv::BitCountOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpBitCount));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitFieldInsertOp>(::mlir::spirv::BitFieldInsertOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpBitFieldInsert));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitFieldSExtractOp>(::mlir::spirv::BitFieldSExtractOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpBitFieldSExtract));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitFieldUExtractOp>(::mlir::spirv::BitFieldUExtractOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpBitFieldUExtract));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitReverseOp>(::mlir::spirv::BitReverseOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpBitReverse));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitcastOp>(::mlir::spirv::BitcastOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpBitcast));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitwiseAndOp>(::mlir::spirv::BitwiseAndOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpBitwiseAnd));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitwiseOrOp>(::mlir::spirv::BitwiseOrOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpBitwiseOr));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitwiseXorOp>(::mlir::spirv::BitwiseXorOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpBitwiseXor));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CLCeilOp>(::mlir::spirv::CLCeilOp op) {
  return processOpWithoutGrammarAttr(op, "OpenCL.std", 12);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CLCosOp>(::mlir::spirv::CLCosOp op) {
  return processOpWithoutGrammarAttr(op, "OpenCL.std", 14);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CLErfOp>(::mlir::spirv::CLErfOp op) {
  return processOpWithoutGrammarAttr(op, "OpenCL.std", 18);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CLExpOp>(::mlir::spirv::CLExpOp op) {
  return processOpWithoutGrammarAttr(op, "OpenCL.std", 19);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CLFAbsOp>(::mlir::spirv::CLFAbsOp op) {
  return processOpWithoutGrammarAttr(op, "OpenCL.std", 23);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CLFloorOp>(::mlir::spirv::CLFloorOp op) {
  return processOpWithoutGrammarAttr(op, "OpenCL.std", 25);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CLFmaOp>(::mlir::spirv::CLFmaOp op) {
  return processOpWithoutGrammarAttr(op, "OpenCL.std", 26);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CLLogOp>(::mlir::spirv::CLLogOp op) {
  return processOpWithoutGrammarAttr(op, "OpenCL.std", 37);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CLPowOp>(::mlir::spirv::CLPowOp op) {
  return processOpWithoutGrammarAttr(op, "OpenCL.std", 48);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CLRoundOp>(::mlir::spirv::CLRoundOp op) {
  return processOpWithoutGrammarAttr(op, "OpenCL.std", 55);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CLRsqrtOp>(::mlir::spirv::CLRsqrtOp op) {
  return processOpWithoutGrammarAttr(op, "OpenCL.std", 56);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CLSAbsOp>(::mlir::spirv::CLSAbsOp op) {
  return processOpWithoutGrammarAttr(op, "OpenCL.std", 141);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CLSinOp>(::mlir::spirv::CLSinOp op) {
  return processOpWithoutGrammarAttr(op, "OpenCL.std", 57);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CLSqrtOp>(::mlir::spirv::CLSqrtOp op) {
  return processOpWithoutGrammarAttr(op, "OpenCL.std", 61);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CLTanhOp>(::mlir::spirv::CLTanhOp op) {
  return processOpWithoutGrammarAttr(op, "OpenCL.std", 63);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CompositeConstructOp>(::mlir::spirv::CompositeConstructOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpCompositeConstruct, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CompositeExtractOp>(::mlir::spirv::CompositeExtractOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op->getAttr("indices")) {
    for (auto attrElem : attr.cast<ArrayAttr>()) {
      operands.push_back(static_cast<uint32_t>(attrElem.cast<IntegerAttr>().getValue().getZExtValue()));
    }
  }
  elidedAttrs.push_back("indices");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpCompositeExtract, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CompositeInsertOp>(::mlir::spirv::CompositeInsertOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op->getAttr("indices")) {
    for (auto attrElem : attr.cast<ArrayAttr>()) {
      operands.push_back(static_cast<uint32_t>(attrElem.cast<IntegerAttr>().getValue().getZExtValue()));
    }
  }
  elidedAttrs.push_back("indices");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpCompositeInsert, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ControlBarrierOp>(::mlir::spirv::ControlBarrierOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  if (auto attr = op->getAttr("execution_scope")) {
    operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
  }
  elidedAttrs.push_back("execution_scope");
  if (auto attr = op->getAttr("memory_scope")) {
    operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
  }
  elidedAttrs.push_back("memory_scope");
  if (auto attr = op->getAttr("memory_semantics")) {
    operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemorySemanticsAttr>().getValue()))));
  }
  elidedAttrs.push_back("memory_semantics");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpControlBarrier, operands);
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ConvertFToSOp>(::mlir::spirv::ConvertFToSOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpConvertFToS));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ConvertFToUOp>(::mlir::spirv::ConvertFToUOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpConvertFToU));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ConvertSToFOp>(::mlir::spirv::ConvertSToFOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpConvertSToF));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ConvertUToFOp>(::mlir::spirv::ConvertUToFOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpConvertUToF));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::EXTAtomicFAddOp>(::mlir::spirv::EXTAtomicFAddOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op->getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemorySemanticsAttr>().getValue()))));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicFAddEXT, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FAddOp>(::mlir::spirv::FAddOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFAdd));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FConvertOp>(::mlir::spirv::FConvertOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFConvert));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FDivOp>(::mlir::spirv::FDivOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFDiv));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FModOp>(::mlir::spirv::FModOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFMod));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FMulOp>(::mlir::spirv::FMulOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFMul));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FNegateOp>(::mlir::spirv::FNegateOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFNegate));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FOrdEqualOp>(::mlir::spirv::FOrdEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFOrdEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FOrdGreaterThanEqualOp>(::mlir::spirv::FOrdGreaterThanEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFOrdGreaterThanEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FOrdGreaterThanOp>(::mlir::spirv::FOrdGreaterThanOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFOrdGreaterThan));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FOrdLessThanEqualOp>(::mlir::spirv::FOrdLessThanEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFOrdLessThanEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FOrdLessThanOp>(::mlir::spirv::FOrdLessThanOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFOrdLessThan));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FOrdNotEqualOp>(::mlir::spirv::FOrdNotEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFOrdNotEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FRemOp>(::mlir::spirv::FRemOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFRem));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FSubOp>(::mlir::spirv::FSubOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFSub));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FUnordEqualOp>(::mlir::spirv::FUnordEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFUnordEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FUnordGreaterThanEqualOp>(::mlir::spirv::FUnordGreaterThanEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFUnordGreaterThanEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FUnordGreaterThanOp>(::mlir::spirv::FUnordGreaterThanOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFUnordGreaterThan));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FUnordLessThanEqualOp>(::mlir::spirv::FUnordLessThanEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFUnordLessThanEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FUnordLessThanOp>(::mlir::spirv::FUnordLessThanOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFUnordLessThan));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FUnordNotEqualOp>(::mlir::spirv::FUnordNotEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpFUnordNotEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLAcosOp>(::mlir::spirv::GLAcosOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 17);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLAsinOp>(::mlir::spirv::GLAsinOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 16);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLAtanOp>(::mlir::spirv::GLAtanOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 18);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLCeilOp>(::mlir::spirv::GLCeilOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 9);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLCosOp>(::mlir::spirv::GLCosOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 14);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLCoshOp>(::mlir::spirv::GLCoshOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 20);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLExpOp>(::mlir::spirv::GLExpOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 27);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLFAbsOp>(::mlir::spirv::GLFAbsOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 4);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLFClampOp>(::mlir::spirv::GLFClampOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 43);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLFMaxOp>(::mlir::spirv::GLFMaxOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 40);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLFMinOp>(::mlir::spirv::GLFMinOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 37);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLFMixOp>(::mlir::spirv::GLFMixOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 46);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLFSignOp>(::mlir::spirv::GLFSignOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 6);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLFindUMsbOp>(::mlir::spirv::GLFindUMsbOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 75);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLFloorOp>(::mlir::spirv::GLFloorOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 8);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLFmaOp>(::mlir::spirv::GLFmaOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 50);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLFrexpStructOp>(::mlir::spirv::GLFrexpStructOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 52);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLInverseSqrtOp>(::mlir::spirv::GLInverseSqrtOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 32);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLLdexpOp>(::mlir::spirv::GLLdexpOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 53);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLLogOp>(::mlir::spirv::GLLogOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 28);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLPowOp>(::mlir::spirv::GLPowOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 26);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLRoundOp>(::mlir::spirv::GLRoundOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 1);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSAbsOp>(::mlir::spirv::GLSAbsOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 5);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSClampOp>(::mlir::spirv::GLSClampOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 45);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSMaxOp>(::mlir::spirv::GLSMaxOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 42);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSMinOp>(::mlir::spirv::GLSMinOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 39);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSSignOp>(::mlir::spirv::GLSSignOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 7);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSinOp>(::mlir::spirv::GLSinOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 13);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSinhOp>(::mlir::spirv::GLSinhOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 19);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSqrtOp>(::mlir::spirv::GLSqrtOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 31);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLTanOp>(::mlir::spirv::GLTanOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 15);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLTanhOp>(::mlir::spirv::GLTanhOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 21);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLUClampOp>(::mlir::spirv::GLUClampOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 44);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLUMaxOp>(::mlir::spirv::GLUMaxOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 41);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLUMinOp>(::mlir::spirv::GLUMinOp op) {
  return processOpWithoutGrammarAttr(op, "GLSL.std.450", 38);
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupBroadcastOp>(::mlir::spirv::GroupBroadcastOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupBroadcast, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformBallotOp>(::mlir::spirv::GroupNonUniformBallotOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformBallot, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformBroadcastOp>(::mlir::spirv::GroupNonUniformBroadcastOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformBroadcast, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformElectOp>(::mlir::spirv::GroupNonUniformElectOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  if (auto attr = op->getAttr("execution_scope")) {
    operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
  }
  elidedAttrs.push_back("execution_scope");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformElect, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformFAddOp>(::mlir::spirv::GroupNonUniformFAddOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op->getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<::mlir::spirv::GroupOperationAttr>().getValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformFAdd, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformFMaxOp>(::mlir::spirv::GroupNonUniformFMaxOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op->getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<::mlir::spirv::GroupOperationAttr>().getValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformFMax, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformFMinOp>(::mlir::spirv::GroupNonUniformFMinOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op->getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<::mlir::spirv::GroupOperationAttr>().getValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformFMin, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformFMulOp>(::mlir::spirv::GroupNonUniformFMulOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op->getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<::mlir::spirv::GroupOperationAttr>().getValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformFMul, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformIAddOp>(::mlir::spirv::GroupNonUniformIAddOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op->getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<::mlir::spirv::GroupOperationAttr>().getValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformIAdd, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformIMulOp>(::mlir::spirv::GroupNonUniformIMulOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op->getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<::mlir::spirv::GroupOperationAttr>().getValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformIMul, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformSMaxOp>(::mlir::spirv::GroupNonUniformSMaxOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op->getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<::mlir::spirv::GroupOperationAttr>().getValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformSMax, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformSMinOp>(::mlir::spirv::GroupNonUniformSMinOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op->getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<::mlir::spirv::GroupOperationAttr>().getValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformSMin, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformShuffleDownOp>(::mlir::spirv::GroupNonUniformShuffleDownOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformShuffleDown, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformShuffleOp>(::mlir::spirv::GroupNonUniformShuffleOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformShuffle, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformShuffleUpOp>(::mlir::spirv::GroupNonUniformShuffleUpOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformShuffleUp, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformShuffleXorOp>(::mlir::spirv::GroupNonUniformShuffleXorOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformShuffleXor, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformUMaxOp>(::mlir::spirv::GroupNonUniformUMaxOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op->getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<::mlir::spirv::GroupOperationAttr>().getValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformUMax, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformUMinOp>(::mlir::spirv::GroupNonUniformUMinOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op->getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<::mlir::spirv::GroupOperationAttr>().getValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformUMin, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::IAddCarryOp>(::mlir::spirv::IAddCarryOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpIAddCarry));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::IAddOp>(::mlir::spirv::IAddOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpIAdd));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::IEqualOp>(::mlir::spirv::IEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpIEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::IMulOp>(::mlir::spirv::IMulOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpIMul));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::INTELJointMatrixLoadOp>(::mlir::spirv::INTELJointMatrixLoadOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("scope");
  }
  {
    if (auto attr = op->getAttr("layout")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MatrixLayoutAttr>().getValue()))));
    }
    elidedAttrs.push_back("layout");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_access")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemoryAccessAttr>().getValue()));
    }
    elidedAttrs.push_back("memory_access");
  }
  {
    if (auto attr = op->getAttr("alignment")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("alignment");
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpJointMatrixLoadINTEL, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::INTELJointMatrixMadOp>(::mlir::spirv::INTELJointMatrixMadOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("scope");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(2)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #2 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpJointMatrixMadINTEL, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::INTELJointMatrixStoreOp>(::mlir::spirv::INTELJointMatrixStoreOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  {
    if (auto attr = op->getAttr("scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
    }
    elidedAttrs.push_back("scope");
  }
  {
    if (auto attr = op->getAttr("layout")) {
      operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MatrixLayoutAttr>().getValue()))));
    }
    elidedAttrs.push_back("layout");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(2)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #2 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("memory_access")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemoryAccessAttr>().getValue()));
    }
    elidedAttrs.push_back("memory_access");
  }
  {
    if (auto attr = op->getAttr("alignment")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("alignment");
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpJointMatrixStoreINTEL, operands);
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::INTELJointMatrixWorkItemLengthOp>(::mlir::spirv::INTELJointMatrixWorkItemLengthOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  if (auto attr = op->getAttr("type")) {
    operands.push_back(static_cast<uint32_t>(getTypeID(attr.cast<TypeAttr>().getValue())));
  }
  elidedAttrs.push_back("type");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpJointMatrixWorkItemLengthINTEL, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::INTELSubgroupBlockReadOp>(::mlir::spirv::INTELSubgroupBlockReadOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSubgroupBlockReadINTEL));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::INTELSubgroupBlockWriteOp>(::mlir::spirv::INTELSubgroupBlockWriteOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSubgroupBlockWriteINTEL));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::INotEqualOp>(::mlir::spirv::INotEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpINotEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ISubBorrowOp>(::mlir::spirv::ISubBorrowOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpISubBorrow));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ISubOp>(::mlir::spirv::ISubOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpISub));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ImageDrefGatherOp>(::mlir::spirv::ImageDrefGatherOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(2)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #2 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op->getAttr("imageoperands")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<::mlir::spirv::ImageOperandsAttr>().getValue()));
    }
    elidedAttrs.push_back("imageoperands");
  }
  {
    for (auto arg : op.getODSOperands(3)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #3 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpImageDrefGather, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ImageOp>(::mlir::spirv::ImageOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpImage));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ImageQuerySizeOp>(::mlir::spirv::ImageQuerySizeOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpImageQuerySize));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::InBoundsPtrAccessChainOp>(::mlir::spirv::InBoundsPtrAccessChainOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpInBoundsPtrAccessChain, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::IsInfOp>(::mlir::spirv::IsInfOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpIsInf));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::IsNanOp>(::mlir::spirv::IsNanOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpIsNan));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::KHRAssumeTrueOp>(::mlir::spirv::KHRAssumeTrueOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpAssumeTrueKHR));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::KHRSubgroupBallotOp>(::mlir::spirv::KHRSubgroupBallotOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSubgroupBallotKHR));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::LoadOp>(::mlir::spirv::LoadOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op->getAttr("memory_access")) {
    operands.push_back(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemoryAccessAttr>().getValue()));
  }
  elidedAttrs.push_back("memory_access");
  if (auto attr = op->getAttr("alignment")) {
    operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
  }
  elidedAttrs.push_back("alignment");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpLoad, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::LogicalAndOp>(::mlir::spirv::LogicalAndOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpLogicalAnd));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::LogicalEqualOp>(::mlir::spirv::LogicalEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpLogicalEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::LogicalNotEqualOp>(::mlir::spirv::LogicalNotEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpLogicalNotEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::LogicalNotOp>(::mlir::spirv::LogicalNotOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpLogicalNot));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::LogicalOrOp>(::mlir::spirv::LogicalOrOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpLogicalOr));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::MatrixTimesMatrixOp>(::mlir::spirv::MatrixTimesMatrixOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpMatrixTimesMatrix));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::MatrixTimesScalarOp>(::mlir::spirv::MatrixTimesScalarOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpMatrixTimesScalar));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::MemoryBarrierOp>(::mlir::spirv::MemoryBarrierOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  if (auto attr = op->getAttr("memory_scope")) {
    operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::ScopeAttr>().getValue()))));
  }
  elidedAttrs.push_back("memory_scope");
  if (auto attr = op->getAttr("memory_semantics")) {
    operands.push_back(prepareConstantInt(op.getLoc(), Builder(op).getI32IntegerAttr(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemorySemanticsAttr>().getValue()))));
  }
  elidedAttrs.push_back("memory_semantics");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpMemoryBarrier, operands);
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::NVCooperativeMatrixLengthOp>(::mlir::spirv::NVCooperativeMatrixLengthOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  if (auto attr = op->getAttr("type")) {
    operands.push_back(static_cast<uint32_t>(getTypeID(attr.cast<TypeAttr>().getValue())));
  }
  elidedAttrs.push_back("type");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpCooperativeMatrixLengthNV, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::NVCooperativeMatrixLoadOp>(::mlir::spirv::NVCooperativeMatrixLoadOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op->getAttr("memory_access")) {
    operands.push_back(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemoryAccessAttr>().getValue()));
  }
  elidedAttrs.push_back("memory_access");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpCooperativeMatrixLoadNV, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::NVCooperativeMatrixMulAddOp>(::mlir::spirv::NVCooperativeMatrixMulAddOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpCooperativeMatrixMulAddNV));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::NVCooperativeMatrixStoreOp>(::mlir::spirv::NVCooperativeMatrixStoreOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op->getAttr("memory_access")) {
    operands.push_back(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemoryAccessAttr>().getValue()));
  }
  elidedAttrs.push_back("memory_access");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpCooperativeMatrixStoreNV, operands);
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::NotOp>(::mlir::spirv::NotOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpNot));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::OrderedOp>(::mlir::spirv::OrderedOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpOrdered));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::PtrAccessChainOp>(::mlir::spirv::PtrAccessChainOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpPtrAccessChain, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ReturnOp>(::mlir::spirv::ReturnOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpReturn));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ReturnValueOp>(::mlir::spirv::ReturnValueOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpReturnValue));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SConvertOp>(::mlir::spirv::SConvertOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSConvert));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SDivOp>(::mlir::spirv::SDivOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSDiv));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SGreaterThanEqualOp>(::mlir::spirv::SGreaterThanEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSGreaterThanEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SGreaterThanOp>(::mlir::spirv::SGreaterThanOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSGreaterThan));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SLessThanEqualOp>(::mlir::spirv::SLessThanEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSLessThanEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SLessThanOp>(::mlir::spirv::SLessThanOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSLessThan));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SModOp>(::mlir::spirv::SModOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSMod));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SNegateOp>(::mlir::spirv::SNegateOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSNegate));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SRemOp>(::mlir::spirv::SRemOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSRem));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SelectOp>(::mlir::spirv::SelectOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpSelect));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ShiftLeftLogicalOp>(::mlir::spirv::ShiftLeftLogicalOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpShiftLeftLogical));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ShiftRightArithmeticOp>(::mlir::spirv::ShiftRightArithmeticOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpShiftRightArithmetic));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ShiftRightLogicalOp>(::mlir::spirv::ShiftRightLogicalOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpShiftRightLogical));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::StoreOp>(::mlir::spirv::StoreOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op->getAttr("memory_access")) {
    operands.push_back(static_cast<uint32_t>(attr.cast<::mlir::spirv::MemoryAccessAttr>().getValue()));
  }
  elidedAttrs.push_back("memory_access");
  if (auto attr = op->getAttr("alignment")) {
    operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
  }
  elidedAttrs.push_back("alignment");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpStore, operands);
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::TransposeOp>(::mlir::spirv::TransposeOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpTranspose));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::UConvertOp>(::mlir::spirv::UConvertOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpUConvert));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::UDivOp>(::mlir::spirv::UDivOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpUDiv));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::UGreaterThanEqualOp>(::mlir::spirv::UGreaterThanEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpUGreaterThanEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::UGreaterThanOp>(::mlir::spirv::UGreaterThanOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpUGreaterThan));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ULessThanEqualOp>(::mlir::spirv::ULessThanEqualOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpULessThanEqual));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ULessThanOp>(::mlir::spirv::ULessThanOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpULessThan));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::UModOp>(::mlir::spirv::UModOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpUMod));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::UnorderedOp>(::mlir::spirv::UnorderedOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpUnordered));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::UnreachableOp>(::mlir::spirv::UnreachableOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpUnreachable));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::VariableOp>(::mlir::spirv::VariableOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op->getAttr("storage_class")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<::mlir::spirv::StorageClassAttr>().getValue()));
    }
    elidedAttrs.push_back("storage_class");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpVariable, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::VectorExtractDynamicOp>(::mlir::spirv::VectorExtractDynamicOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpVectorExtractDynamic));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::VectorInsertDynamicOp>(::mlir::spirv::VectorInsertDynamicOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpVectorInsertDynamic));
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::VectorShuffleOp>(::mlir::spirv::VectorShuffleOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op->getAttr("components")) {
    for (auto attrElem : attr.cast<ArrayAttr>()) {
      operands.push_back(static_cast<uint32_t>(attrElem.cast<IntegerAttr>().getValue().getZExtValue()));
    }
  }
  elidedAttrs.push_back("components");
  (void)emitDebugLine(functionBody, op.getLoc());
  (void)encodeInstructionInto(functionBody, spirv::Opcode::OpVectorShuffle, operands);
  for (auto attr : op->getAttrs()) {
    if (llvm::is_contained(elidedAttrs, attr.getName())) {      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::VectorTimesScalarOp>(::mlir::spirv::VectorTimesScalarOp op) {
  return processOpWithoutGrammarAttr(op, "", static_cast<uint32_t>(spirv::Opcode::OpVectorTimesScalar));
}

LogicalResult Serializer::dispatchToAutogenSerialization(Operation *op) {
  if (isa<::mlir::spirv::AccessChainOp>(op)) {
    return processOp(cast<::mlir::spirv::AccessChainOp>(op));
  }
  if (isa<::mlir::spirv::AtomicAndOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicAndOp>(op));
  }
  if (isa<::mlir::spirv::AtomicCompareExchangeOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicCompareExchangeOp>(op));
  }
  if (isa<::mlir::spirv::AtomicCompareExchangeWeakOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicCompareExchangeWeakOp>(op));
  }
  if (isa<::mlir::spirv::AtomicExchangeOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicExchangeOp>(op));
  }
  if (isa<::mlir::spirv::AtomicIAddOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicIAddOp>(op));
  }
  if (isa<::mlir::spirv::AtomicIDecrementOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicIDecrementOp>(op));
  }
  if (isa<::mlir::spirv::AtomicIIncrementOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicIIncrementOp>(op));
  }
  if (isa<::mlir::spirv::AtomicISubOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicISubOp>(op));
  }
  if (isa<::mlir::spirv::AtomicOrOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicOrOp>(op));
  }
  if (isa<::mlir::spirv::AtomicSMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicSMaxOp>(op));
  }
  if (isa<::mlir::spirv::AtomicSMinOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicSMinOp>(op));
  }
  if (isa<::mlir::spirv::AtomicUMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicUMaxOp>(op));
  }
  if (isa<::mlir::spirv::AtomicUMinOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicUMinOp>(op));
  }
  if (isa<::mlir::spirv::AtomicXorOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicXorOp>(op));
  }
  if (isa<::mlir::spirv::BitCountOp>(op)) {
    return processOp(cast<::mlir::spirv::BitCountOp>(op));
  }
  if (isa<::mlir::spirv::BitFieldInsertOp>(op)) {
    return processOp(cast<::mlir::spirv::BitFieldInsertOp>(op));
  }
  if (isa<::mlir::spirv::BitFieldSExtractOp>(op)) {
    return processOp(cast<::mlir::spirv::BitFieldSExtractOp>(op));
  }
  if (isa<::mlir::spirv::BitFieldUExtractOp>(op)) {
    return processOp(cast<::mlir::spirv::BitFieldUExtractOp>(op));
  }
  if (isa<::mlir::spirv::BitReverseOp>(op)) {
    return processOp(cast<::mlir::spirv::BitReverseOp>(op));
  }
  if (isa<::mlir::spirv::BitcastOp>(op)) {
    return processOp(cast<::mlir::spirv::BitcastOp>(op));
  }
  if (isa<::mlir::spirv::BitwiseAndOp>(op)) {
    return processOp(cast<::mlir::spirv::BitwiseAndOp>(op));
  }
  if (isa<::mlir::spirv::BitwiseOrOp>(op)) {
    return processOp(cast<::mlir::spirv::BitwiseOrOp>(op));
  }
  if (isa<::mlir::spirv::BitwiseXorOp>(op)) {
    return processOp(cast<::mlir::spirv::BitwiseXorOp>(op));
  }
  if (isa<::mlir::spirv::BranchConditionalOp>(op)) {
    return processOp(cast<::mlir::spirv::BranchConditionalOp>(op));
  }
  if (isa<::mlir::spirv::BranchOp>(op)) {
    return processOp(cast<::mlir::spirv::BranchOp>(op));
  }
  if (isa<::mlir::spirv::CLCeilOp>(op)) {
    return processOp(cast<::mlir::spirv::CLCeilOp>(op));
  }
  if (isa<::mlir::spirv::CLCosOp>(op)) {
    return processOp(cast<::mlir::spirv::CLCosOp>(op));
  }
  if (isa<::mlir::spirv::CLErfOp>(op)) {
    return processOp(cast<::mlir::spirv::CLErfOp>(op));
  }
  if (isa<::mlir::spirv::CLExpOp>(op)) {
    return processOp(cast<::mlir::spirv::CLExpOp>(op));
  }
  if (isa<::mlir::spirv::CLFAbsOp>(op)) {
    return processOp(cast<::mlir::spirv::CLFAbsOp>(op));
  }
  if (isa<::mlir::spirv::CLFloorOp>(op)) {
    return processOp(cast<::mlir::spirv::CLFloorOp>(op));
  }
  if (isa<::mlir::spirv::CLFmaOp>(op)) {
    return processOp(cast<::mlir::spirv::CLFmaOp>(op));
  }
  if (isa<::mlir::spirv::CLLogOp>(op)) {
    return processOp(cast<::mlir::spirv::CLLogOp>(op));
  }
  if (isa<::mlir::spirv::CLPowOp>(op)) {
    return processOp(cast<::mlir::spirv::CLPowOp>(op));
  }
  if (isa<::mlir::spirv::CLRoundOp>(op)) {
    return processOp(cast<::mlir::spirv::CLRoundOp>(op));
  }
  if (isa<::mlir::spirv::CLRsqrtOp>(op)) {
    return processOp(cast<::mlir::spirv::CLRsqrtOp>(op));
  }
  if (isa<::mlir::spirv::CLSAbsOp>(op)) {
    return processOp(cast<::mlir::spirv::CLSAbsOp>(op));
  }
  if (isa<::mlir::spirv::CLSinOp>(op)) {
    return processOp(cast<::mlir::spirv::CLSinOp>(op));
  }
  if (isa<::mlir::spirv::CLSqrtOp>(op)) {
    return processOp(cast<::mlir::spirv::CLSqrtOp>(op));
  }
  if (isa<::mlir::spirv::CLTanhOp>(op)) {
    return processOp(cast<::mlir::spirv::CLTanhOp>(op));
  }
  if (isa<::mlir::spirv::CompositeConstructOp>(op)) {
    return processOp(cast<::mlir::spirv::CompositeConstructOp>(op));
  }
  if (isa<::mlir::spirv::CompositeExtractOp>(op)) {
    return processOp(cast<::mlir::spirv::CompositeExtractOp>(op));
  }
  if (isa<::mlir::spirv::CompositeInsertOp>(op)) {
    return processOp(cast<::mlir::spirv::CompositeInsertOp>(op));
  }
  if (isa<::mlir::spirv::ControlBarrierOp>(op)) {
    return processOp(cast<::mlir::spirv::ControlBarrierOp>(op));
  }
  if (isa<::mlir::spirv::ConvertFToSOp>(op)) {
    return processOp(cast<::mlir::spirv::ConvertFToSOp>(op));
  }
  if (isa<::mlir::spirv::ConvertFToUOp>(op)) {
    return processOp(cast<::mlir::spirv::ConvertFToUOp>(op));
  }
  if (isa<::mlir::spirv::ConvertSToFOp>(op)) {
    return processOp(cast<::mlir::spirv::ConvertSToFOp>(op));
  }
  if (isa<::mlir::spirv::ConvertUToFOp>(op)) {
    return processOp(cast<::mlir::spirv::ConvertUToFOp>(op));
  }
  if (isa<::mlir::spirv::CopyMemoryOp>(op)) {
    return processOp(cast<::mlir::spirv::CopyMemoryOp>(op));
  }
  if (isa<::mlir::spirv::EXTAtomicFAddOp>(op)) {
    return processOp(cast<::mlir::spirv::EXTAtomicFAddOp>(op));
  }
  if (isa<::mlir::spirv::EntryPointOp>(op)) {
    return processOp(cast<::mlir::spirv::EntryPointOp>(op));
  }
  if (isa<::mlir::spirv::ExecutionModeOp>(op)) {
    return processOp(cast<::mlir::spirv::ExecutionModeOp>(op));
  }
  if (isa<::mlir::spirv::FAddOp>(op)) {
    return processOp(cast<::mlir::spirv::FAddOp>(op));
  }
  if (isa<::mlir::spirv::FConvertOp>(op)) {
    return processOp(cast<::mlir::spirv::FConvertOp>(op));
  }
  if (isa<::mlir::spirv::FDivOp>(op)) {
    return processOp(cast<::mlir::spirv::FDivOp>(op));
  }
  if (isa<::mlir::spirv::FModOp>(op)) {
    return processOp(cast<::mlir::spirv::FModOp>(op));
  }
  if (isa<::mlir::spirv::FMulOp>(op)) {
    return processOp(cast<::mlir::spirv::FMulOp>(op));
  }
  if (isa<::mlir::spirv::FNegateOp>(op)) {
    return processOp(cast<::mlir::spirv::FNegateOp>(op));
  }
  if (isa<::mlir::spirv::FOrdEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FOrdEqualOp>(op));
  }
  if (isa<::mlir::spirv::FOrdGreaterThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FOrdGreaterThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::FOrdGreaterThanOp>(op)) {
    return processOp(cast<::mlir::spirv::FOrdGreaterThanOp>(op));
  }
  if (isa<::mlir::spirv::FOrdLessThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FOrdLessThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::FOrdLessThanOp>(op)) {
    return processOp(cast<::mlir::spirv::FOrdLessThanOp>(op));
  }
  if (isa<::mlir::spirv::FOrdNotEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FOrdNotEqualOp>(op));
  }
  if (isa<::mlir::spirv::FRemOp>(op)) {
    return processOp(cast<::mlir::spirv::FRemOp>(op));
  }
  if (isa<::mlir::spirv::FSubOp>(op)) {
    return processOp(cast<::mlir::spirv::FSubOp>(op));
  }
  if (isa<::mlir::spirv::FUnordEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FUnordEqualOp>(op));
  }
  if (isa<::mlir::spirv::FUnordGreaterThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FUnordGreaterThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::FUnordGreaterThanOp>(op)) {
    return processOp(cast<::mlir::spirv::FUnordGreaterThanOp>(op));
  }
  if (isa<::mlir::spirv::FUnordLessThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FUnordLessThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::FUnordLessThanOp>(op)) {
    return processOp(cast<::mlir::spirv::FUnordLessThanOp>(op));
  }
  if (isa<::mlir::spirv::FUnordNotEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FUnordNotEqualOp>(op));
  }
  if (isa<::mlir::spirv::FunctionCallOp>(op)) {
    return processOp(cast<::mlir::spirv::FunctionCallOp>(op));
  }
  if (isa<::mlir::spirv::GLAcosOp>(op)) {
    return processOp(cast<::mlir::spirv::GLAcosOp>(op));
  }
  if (isa<::mlir::spirv::GLAsinOp>(op)) {
    return processOp(cast<::mlir::spirv::GLAsinOp>(op));
  }
  if (isa<::mlir::spirv::GLAtanOp>(op)) {
    return processOp(cast<::mlir::spirv::GLAtanOp>(op));
  }
  if (isa<::mlir::spirv::GLCeilOp>(op)) {
    return processOp(cast<::mlir::spirv::GLCeilOp>(op));
  }
  if (isa<::mlir::spirv::GLCosOp>(op)) {
    return processOp(cast<::mlir::spirv::GLCosOp>(op));
  }
  if (isa<::mlir::spirv::GLCoshOp>(op)) {
    return processOp(cast<::mlir::spirv::GLCoshOp>(op));
  }
  if (isa<::mlir::spirv::GLExpOp>(op)) {
    return processOp(cast<::mlir::spirv::GLExpOp>(op));
  }
  if (isa<::mlir::spirv::GLFAbsOp>(op)) {
    return processOp(cast<::mlir::spirv::GLFAbsOp>(op));
  }
  if (isa<::mlir::spirv::GLFClampOp>(op)) {
    return processOp(cast<::mlir::spirv::GLFClampOp>(op));
  }
  if (isa<::mlir::spirv::GLFMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::GLFMaxOp>(op));
  }
  if (isa<::mlir::spirv::GLFMinOp>(op)) {
    return processOp(cast<::mlir::spirv::GLFMinOp>(op));
  }
  if (isa<::mlir::spirv::GLFMixOp>(op)) {
    return processOp(cast<::mlir::spirv::GLFMixOp>(op));
  }
  if (isa<::mlir::spirv::GLFSignOp>(op)) {
    return processOp(cast<::mlir::spirv::GLFSignOp>(op));
  }
  if (isa<::mlir::spirv::GLFindUMsbOp>(op)) {
    return processOp(cast<::mlir::spirv::GLFindUMsbOp>(op));
  }
  if (isa<::mlir::spirv::GLFloorOp>(op)) {
    return processOp(cast<::mlir::spirv::GLFloorOp>(op));
  }
  if (isa<::mlir::spirv::GLFmaOp>(op)) {
    return processOp(cast<::mlir::spirv::GLFmaOp>(op));
  }
  if (isa<::mlir::spirv::GLFrexpStructOp>(op)) {
    return processOp(cast<::mlir::spirv::GLFrexpStructOp>(op));
  }
  if (isa<::mlir::spirv::GLInverseSqrtOp>(op)) {
    return processOp(cast<::mlir::spirv::GLInverseSqrtOp>(op));
  }
  if (isa<::mlir::spirv::GLLdexpOp>(op)) {
    return processOp(cast<::mlir::spirv::GLLdexpOp>(op));
  }
  if (isa<::mlir::spirv::GLLogOp>(op)) {
    return processOp(cast<::mlir::spirv::GLLogOp>(op));
  }
  if (isa<::mlir::spirv::GLPowOp>(op)) {
    return processOp(cast<::mlir::spirv::GLPowOp>(op));
  }
  if (isa<::mlir::spirv::GLRoundOp>(op)) {
    return processOp(cast<::mlir::spirv::GLRoundOp>(op));
  }
  if (isa<::mlir::spirv::GLSAbsOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSAbsOp>(op));
  }
  if (isa<::mlir::spirv::GLSClampOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSClampOp>(op));
  }
  if (isa<::mlir::spirv::GLSMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSMaxOp>(op));
  }
  if (isa<::mlir::spirv::GLSMinOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSMinOp>(op));
  }
  if (isa<::mlir::spirv::GLSSignOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSSignOp>(op));
  }
  if (isa<::mlir::spirv::GLSinOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSinOp>(op));
  }
  if (isa<::mlir::spirv::GLSinhOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSinhOp>(op));
  }
  if (isa<::mlir::spirv::GLSqrtOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSqrtOp>(op));
  }
  if (isa<::mlir::spirv::GLTanOp>(op)) {
    return processOp(cast<::mlir::spirv::GLTanOp>(op));
  }
  if (isa<::mlir::spirv::GLTanhOp>(op)) {
    return processOp(cast<::mlir::spirv::GLTanhOp>(op));
  }
  if (isa<::mlir::spirv::GLUClampOp>(op)) {
    return processOp(cast<::mlir::spirv::GLUClampOp>(op));
  }
  if (isa<::mlir::spirv::GLUMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::GLUMaxOp>(op));
  }
  if (isa<::mlir::spirv::GLUMinOp>(op)) {
    return processOp(cast<::mlir::spirv::GLUMinOp>(op));
  }
  if (isa<::mlir::spirv::GroupBroadcastOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupBroadcastOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformBallotOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformBallotOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformBroadcastOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformBroadcastOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformElectOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformElectOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformFAddOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformFAddOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformFMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformFMaxOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformFMinOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformFMinOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformFMulOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformFMulOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformIAddOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformIAddOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformIMulOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformIMulOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformSMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformSMaxOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformSMinOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformSMinOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformShuffleDownOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformShuffleDownOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformShuffleOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformShuffleOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformShuffleUpOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformShuffleUpOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformShuffleXorOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformShuffleXorOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformUMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformUMaxOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformUMinOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformUMinOp>(op));
  }
  if (isa<::mlir::spirv::IAddCarryOp>(op)) {
    return processOp(cast<::mlir::spirv::IAddCarryOp>(op));
  }
  if (isa<::mlir::spirv::IAddOp>(op)) {
    return processOp(cast<::mlir::spirv::IAddOp>(op));
  }
  if (isa<::mlir::spirv::IEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::IEqualOp>(op));
  }
  if (isa<::mlir::spirv::IMulOp>(op)) {
    return processOp(cast<::mlir::spirv::IMulOp>(op));
  }
  if (isa<::mlir::spirv::INTELJointMatrixLoadOp>(op)) {
    return processOp(cast<::mlir::spirv::INTELJointMatrixLoadOp>(op));
  }
  if (isa<::mlir::spirv::INTELJointMatrixMadOp>(op)) {
    return processOp(cast<::mlir::spirv::INTELJointMatrixMadOp>(op));
  }
  if (isa<::mlir::spirv::INTELJointMatrixStoreOp>(op)) {
    return processOp(cast<::mlir::spirv::INTELJointMatrixStoreOp>(op));
  }
  if (isa<::mlir::spirv::INTELJointMatrixWorkItemLengthOp>(op)) {
    return processOp(cast<::mlir::spirv::INTELJointMatrixWorkItemLengthOp>(op));
  }
  if (isa<::mlir::spirv::INTELSubgroupBlockReadOp>(op)) {
    return processOp(cast<::mlir::spirv::INTELSubgroupBlockReadOp>(op));
  }
  if (isa<::mlir::spirv::INTELSubgroupBlockWriteOp>(op)) {
    return processOp(cast<::mlir::spirv::INTELSubgroupBlockWriteOp>(op));
  }
  if (isa<::mlir::spirv::INotEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::INotEqualOp>(op));
  }
  if (isa<::mlir::spirv::ISubBorrowOp>(op)) {
    return processOp(cast<::mlir::spirv::ISubBorrowOp>(op));
  }
  if (isa<::mlir::spirv::ISubOp>(op)) {
    return processOp(cast<::mlir::spirv::ISubOp>(op));
  }
  if (isa<::mlir::spirv::ImageDrefGatherOp>(op)) {
    return processOp(cast<::mlir::spirv::ImageDrefGatherOp>(op));
  }
  if (isa<::mlir::spirv::ImageOp>(op)) {
    return processOp(cast<::mlir::spirv::ImageOp>(op));
  }
  if (isa<::mlir::spirv::ImageQuerySizeOp>(op)) {
    return processOp(cast<::mlir::spirv::ImageQuerySizeOp>(op));
  }
  if (isa<::mlir::spirv::InBoundsPtrAccessChainOp>(op)) {
    return processOp(cast<::mlir::spirv::InBoundsPtrAccessChainOp>(op));
  }
  if (isa<::mlir::spirv::IsInfOp>(op)) {
    return processOp(cast<::mlir::spirv::IsInfOp>(op));
  }
  if (isa<::mlir::spirv::IsNanOp>(op)) {
    return processOp(cast<::mlir::spirv::IsNanOp>(op));
  }
  if (isa<::mlir::spirv::KHRAssumeTrueOp>(op)) {
    return processOp(cast<::mlir::spirv::KHRAssumeTrueOp>(op));
  }
  if (isa<::mlir::spirv::KHRSubgroupBallotOp>(op)) {
    return processOp(cast<::mlir::spirv::KHRSubgroupBallotOp>(op));
  }
  if (isa<::mlir::spirv::LoadOp>(op)) {
    return processOp(cast<::mlir::spirv::LoadOp>(op));
  }
  if (isa<::mlir::spirv::LogicalAndOp>(op)) {
    return processOp(cast<::mlir::spirv::LogicalAndOp>(op));
  }
  if (isa<::mlir::spirv::LogicalEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::LogicalEqualOp>(op));
  }
  if (isa<::mlir::spirv::LogicalNotEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::LogicalNotEqualOp>(op));
  }
  if (isa<::mlir::spirv::LogicalNotOp>(op)) {
    return processOp(cast<::mlir::spirv::LogicalNotOp>(op));
  }
  if (isa<::mlir::spirv::LogicalOrOp>(op)) {
    return processOp(cast<::mlir::spirv::LogicalOrOp>(op));
  }
  if (isa<::mlir::spirv::MatrixTimesMatrixOp>(op)) {
    return processOp(cast<::mlir::spirv::MatrixTimesMatrixOp>(op));
  }
  if (isa<::mlir::spirv::MatrixTimesScalarOp>(op)) {
    return processOp(cast<::mlir::spirv::MatrixTimesScalarOp>(op));
  }
  if (isa<::mlir::spirv::MemoryBarrierOp>(op)) {
    return processOp(cast<::mlir::spirv::MemoryBarrierOp>(op));
  }
  if (isa<::mlir::spirv::NVCooperativeMatrixLengthOp>(op)) {
    return processOp(cast<::mlir::spirv::NVCooperativeMatrixLengthOp>(op));
  }
  if (isa<::mlir::spirv::NVCooperativeMatrixLoadOp>(op)) {
    return processOp(cast<::mlir::spirv::NVCooperativeMatrixLoadOp>(op));
  }
  if (isa<::mlir::spirv::NVCooperativeMatrixMulAddOp>(op)) {
    return processOp(cast<::mlir::spirv::NVCooperativeMatrixMulAddOp>(op));
  }
  if (isa<::mlir::spirv::NVCooperativeMatrixStoreOp>(op)) {
    return processOp(cast<::mlir::spirv::NVCooperativeMatrixStoreOp>(op));
  }
  if (isa<::mlir::spirv::NotOp>(op)) {
    return processOp(cast<::mlir::spirv::NotOp>(op));
  }
  if (isa<::mlir::spirv::OrderedOp>(op)) {
    return processOp(cast<::mlir::spirv::OrderedOp>(op));
  }
  if (isa<::mlir::spirv::PtrAccessChainOp>(op)) {
    return processOp(cast<::mlir::spirv::PtrAccessChainOp>(op));
  }
  if (isa<::mlir::spirv::ReturnOp>(op)) {
    return processOp(cast<::mlir::spirv::ReturnOp>(op));
  }
  if (isa<::mlir::spirv::ReturnValueOp>(op)) {
    return processOp(cast<::mlir::spirv::ReturnValueOp>(op));
  }
  if (isa<::mlir::spirv::SConvertOp>(op)) {
    return processOp(cast<::mlir::spirv::SConvertOp>(op));
  }
  if (isa<::mlir::spirv::SDivOp>(op)) {
    return processOp(cast<::mlir::spirv::SDivOp>(op));
  }
  if (isa<::mlir::spirv::SGreaterThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::SGreaterThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::SGreaterThanOp>(op)) {
    return processOp(cast<::mlir::spirv::SGreaterThanOp>(op));
  }
  if (isa<::mlir::spirv::SLessThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::SLessThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::SLessThanOp>(op)) {
    return processOp(cast<::mlir::spirv::SLessThanOp>(op));
  }
  if (isa<::mlir::spirv::SModOp>(op)) {
    return processOp(cast<::mlir::spirv::SModOp>(op));
  }
  if (isa<::mlir::spirv::SNegateOp>(op)) {
    return processOp(cast<::mlir::spirv::SNegateOp>(op));
  }
  if (isa<::mlir::spirv::SRemOp>(op)) {
    return processOp(cast<::mlir::spirv::SRemOp>(op));
  }
  if (isa<::mlir::spirv::SelectOp>(op)) {
    return processOp(cast<::mlir::spirv::SelectOp>(op));
  }
  if (isa<::mlir::spirv::ShiftLeftLogicalOp>(op)) {
    return processOp(cast<::mlir::spirv::ShiftLeftLogicalOp>(op));
  }
  if (isa<::mlir::spirv::ShiftRightArithmeticOp>(op)) {
    return processOp(cast<::mlir::spirv::ShiftRightArithmeticOp>(op));
  }
  if (isa<::mlir::spirv::ShiftRightLogicalOp>(op)) {
    return processOp(cast<::mlir::spirv::ShiftRightLogicalOp>(op));
  }
  if (isa<::mlir::spirv::StoreOp>(op)) {
    return processOp(cast<::mlir::spirv::StoreOp>(op));
  }
  if (isa<::mlir::spirv::TransposeOp>(op)) {
    return processOp(cast<::mlir::spirv::TransposeOp>(op));
  }
  if (isa<::mlir::spirv::UConvertOp>(op)) {
    return processOp(cast<::mlir::spirv::UConvertOp>(op));
  }
  if (isa<::mlir::spirv::UDivOp>(op)) {
    return processOp(cast<::mlir::spirv::UDivOp>(op));
  }
  if (isa<::mlir::spirv::UGreaterThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::UGreaterThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::UGreaterThanOp>(op)) {
    return processOp(cast<::mlir::spirv::UGreaterThanOp>(op));
  }
  if (isa<::mlir::spirv::ULessThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::ULessThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::ULessThanOp>(op)) {
    return processOp(cast<::mlir::spirv::ULessThanOp>(op));
  }
  if (isa<::mlir::spirv::UModOp>(op)) {
    return processOp(cast<::mlir::spirv::UModOp>(op));
  }
  if (isa<::mlir::spirv::UnorderedOp>(op)) {
    return processOp(cast<::mlir::spirv::UnorderedOp>(op));
  }
  if (isa<::mlir::spirv::UnreachableOp>(op)) {
    return processOp(cast<::mlir::spirv::UnreachableOp>(op));
  }
  if (isa<::mlir::spirv::VariableOp>(op)) {
    return processOp(cast<::mlir::spirv::VariableOp>(op));
  }
  if (isa<::mlir::spirv::VectorExtractDynamicOp>(op)) {
    return processOp(cast<::mlir::spirv::VectorExtractDynamicOp>(op));
  }
  if (isa<::mlir::spirv::VectorInsertDynamicOp>(op)) {
    return processOp(cast<::mlir::spirv::VectorInsertDynamicOp>(op));
  }
  if (isa<::mlir::spirv::VectorShuffleOp>(op)) {
    return processOp(cast<::mlir::spirv::VectorShuffleOp>(op));
  }
  if (isa<::mlir::spirv::VectorTimesScalarOp>(op)) {
    return processOp(cast<::mlir::spirv::VectorTimesScalarOp>(op));
  }
  return op->emitError("unhandled operation serialization");
}

#endif // GET_SERIALIZATION_FNS

#ifdef GET_DESERIALIZATION_FNS

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AccessChainOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AccessChainOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AccessChainOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AccessChainOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AccessChainOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicAndOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicAndOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicAndOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", opBuilder.getAttr<::mlir::spirv::MemorySemanticsAttr>(static_cast<::mlir::spirv::MemorySemantics>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicAndOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicAndOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicCompareExchangeOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicCompareExchangeOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicCompareExchangeOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("equal_semantics", opBuilder.getAttr<::mlir::spirv::MemorySemanticsAttr>(static_cast<::mlir::spirv::MemorySemantics>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("unequal_semantics", opBuilder.getAttr<::mlir::spirv::MemorySemanticsAttr>(static_cast<::mlir::spirv::MemorySemantics>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicCompareExchangeOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicCompareExchangeOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicCompareExchangeWeakOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicCompareExchangeWeakOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicCompareExchangeWeakOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("equal_semantics", opBuilder.getAttr<::mlir::spirv::MemorySemanticsAttr>(static_cast<::mlir::spirv::MemorySemantics>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("unequal_semantics", opBuilder.getAttr<::mlir::spirv::MemorySemanticsAttr>(static_cast<::mlir::spirv::MemorySemantics>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicCompareExchangeWeakOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicCompareExchangeWeakOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicExchangeOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicExchangeOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicExchangeOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", opBuilder.getAttr<::mlir::spirv::MemorySemanticsAttr>(static_cast<::mlir::spirv::MemorySemantics>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicExchangeOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicExchangeOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicIAddOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicIAddOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicIAddOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", opBuilder.getAttr<::mlir::spirv::MemorySemanticsAttr>(static_cast<::mlir::spirv::MemorySemantics>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicIAddOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicIAddOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicIDecrementOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicIDecrementOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicIDecrementOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", opBuilder.getAttr<::mlir::spirv::MemorySemanticsAttr>(static_cast<::mlir::spirv::MemorySemantics>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicIDecrementOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicIDecrementOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicIIncrementOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicIIncrementOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicIIncrementOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", opBuilder.getAttr<::mlir::spirv::MemorySemanticsAttr>(static_cast<::mlir::spirv::MemorySemantics>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicIIncrementOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicIIncrementOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicISubOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicISubOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicISubOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", opBuilder.getAttr<::mlir::spirv::MemorySemanticsAttr>(static_cast<::mlir::spirv::MemorySemantics>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicISubOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicISubOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicOrOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicOrOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicOrOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", opBuilder.getAttr<::mlir::spirv::MemorySemanticsAttr>(static_cast<::mlir::spirv::MemorySemantics>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicOrOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicOrOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicSMaxOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicSMaxOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicSMaxOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", opBuilder.getAttr<::mlir::spirv::MemorySemanticsAttr>(static_cast<::mlir::spirv::MemorySemantics>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicSMaxOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicSMaxOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicSMinOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicSMinOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicSMinOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", opBuilder.getAttr<::mlir::spirv::MemorySemanticsAttr>(static_cast<::mlir::spirv::MemorySemantics>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicSMinOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicSMinOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicUMaxOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicUMaxOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicUMaxOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", opBuilder.getAttr<::mlir::spirv::MemorySemanticsAttr>(static_cast<::mlir::spirv::MemorySemantics>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicUMaxOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicUMaxOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicUMinOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicUMinOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicUMinOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", opBuilder.getAttr<::mlir::spirv::MemorySemanticsAttr>(static_cast<::mlir::spirv::MemorySemantics>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicUMinOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicUMinOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicXorOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicXorOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicXorOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", opBuilder.getAttr<::mlir::spirv::MemorySemanticsAttr>(static_cast<::mlir::spirv::MemorySemantics>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicXorOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicXorOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitCountOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.BitCount", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitFieldInsertOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.BitFieldInsert", true, 4);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitFieldSExtractOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.BitFieldSExtract", true, 3);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitFieldUExtractOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.BitFieldUExtract", true, 3);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitReverseOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.BitReverse", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitcastOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.Bitcast", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitwiseAndOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.BitwiseAnd", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitwiseOrOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.BitwiseOr", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitwiseXorOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.BitwiseXor", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CLCeilOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.CL.ceil", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CLCosOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.CL.cos", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CLErfOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.CL.erf", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CLExpOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.CL.exp", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CLFAbsOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.CL.fabs", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CLFloorOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.CL.floor", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CLFmaOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.CL.fma", true, 3);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CLLogOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.CL.log", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CLPowOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.CL.pow", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CLRoundOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.CL.round", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CLRsqrtOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.CL.rsqrt", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CLSAbsOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.CL.s_abs", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CLSinOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.CL.sin", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CLSqrtOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.CL.sqrt", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CLTanhOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.CL.tanh", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CompositeConstructOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::CompositeConstructOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::CompositeConstructOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::CompositeConstructOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::CompositeConstructOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CompositeExtractOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::CompositeExtractOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::CompositeExtractOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    SmallVector<Attribute, 4> attrListElems;
    while (wordIndex < words.size()) {
      attrListElems.push_back(opBuilder.getI32IntegerAttr(words[wordIndex++]));
    }
    attributes.push_back(opBuilder.getNamedAttr("indices", opBuilder.getArrayAttr(attrListElems)));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::CompositeExtractOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::CompositeExtractOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CompositeInsertOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::CompositeInsertOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::CompositeInsertOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    SmallVector<Attribute, 4> attrListElems;
    while (wordIndex < words.size()) {
      attrListElems.push_back(opBuilder.getI32IntegerAttr(words[wordIndex++]));
    }
    attributes.push_back(opBuilder.getNamedAttr("indices", opBuilder.getArrayAttr(attrListElems)));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::CompositeInsertOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::CompositeInsertOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ControlBarrierOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_semantics", opBuilder.getAttr<::mlir::spirv::MemorySemanticsAttr>(static_cast<::mlir::spirv::MemorySemantics>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::ControlBarrierOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::ControlBarrierOp>(loc, resultTypes, operands, attributes); (void)op;
  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ConvertFToSOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ConvertFToS", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ConvertFToUOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ConvertFToU", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ConvertSToFOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ConvertSToF", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ConvertUToFOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ConvertUToF", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::EXTAtomicFAddOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::EXTAtomicFAddOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::EXTAtomicFAddOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", opBuilder.getAttr<::mlir::spirv::MemorySemanticsAttr>(static_cast<::mlir::spirv::MemorySemantics>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::EXTAtomicFAddOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::EXTAtomicFAddOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FAddOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FAdd", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FConvertOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FConvert", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FDivOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FDiv", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FModOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FMod", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FMulOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FMul", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FNegateOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FNegate", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FOrdEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FOrdEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FOrdGreaterThanEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FOrdGreaterThanEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FOrdGreaterThanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FOrdGreaterThan", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FOrdLessThanEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FOrdLessThanEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FOrdLessThanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FOrdLessThan", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FOrdNotEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FOrdNotEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FRemOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FRem", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FSubOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FSub", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FUnordEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FUnordEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FUnordGreaterThanEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FUnordGreaterThanEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FUnordGreaterThanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FUnordGreaterThan", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FUnordLessThanEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FUnordLessThanEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FUnordLessThanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FUnordLessThan", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FUnordNotEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.FUnordNotEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLAcosOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.Acos", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLAsinOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.Asin", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLAtanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.Atan", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLCeilOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.Ceil", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLCosOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.Cos", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLCoshOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.Cosh", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLExpOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.Exp", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLFAbsOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.FAbs", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLFClampOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.FClamp", true, 3);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLFMaxOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.FMax", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLFMinOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.FMin", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLFMixOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.FMix", true, 3);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLFSignOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.FSign", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLFindUMsbOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.FindUMsb", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLFloorOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.Floor", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLFmaOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.Fma", true, 3);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLFrexpStructOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.FrexpStruct", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLInverseSqrtOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.InverseSqrt", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLLdexpOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.Ldexp", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLLogOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.Log", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLPowOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.Pow", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLRoundOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.Round", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSAbsOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.SAbs", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSClampOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.SClamp", true, 3);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSMaxOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.SMax", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSMinOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.SMin", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSSignOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.SSign", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSinOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.Sin", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSinhOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.Sinh", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSqrtOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.Sqrt", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLTanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.Tan", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLTanhOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.Tanh", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLUClampOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.UClamp", true, 3);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLUMaxOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.UMax", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLUMinOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.GL.UMin", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupBroadcastOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupBroadcastOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupBroadcastOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupBroadcastOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupBroadcastOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformBallotOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformBallotOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformBallotOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformBallotOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformBallotOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformBroadcastOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformBroadcastOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformBroadcastOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformBroadcastOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformBroadcastOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformElectOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformElectOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformElectOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformElectOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformElectOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformFAddOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformFAddOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformFAddOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
      attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getAttr<::mlir::spirv::GroupOperationAttr>(static_cast<::mlir::spirv::GroupOperation>(words[wordIndex++]))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformFAddOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformFAddOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformFMaxOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformFMaxOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformFMaxOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
      attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getAttr<::mlir::spirv::GroupOperationAttr>(static_cast<::mlir::spirv::GroupOperation>(words[wordIndex++]))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformFMaxOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformFMaxOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformFMinOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformFMinOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformFMinOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
      attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getAttr<::mlir::spirv::GroupOperationAttr>(static_cast<::mlir::spirv::GroupOperation>(words[wordIndex++]))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformFMinOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformFMinOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformFMulOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformFMulOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformFMulOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
      attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getAttr<::mlir::spirv::GroupOperationAttr>(static_cast<::mlir::spirv::GroupOperation>(words[wordIndex++]))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformFMulOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformFMulOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformIAddOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformIAddOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformIAddOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
      attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getAttr<::mlir::spirv::GroupOperationAttr>(static_cast<::mlir::spirv::GroupOperation>(words[wordIndex++]))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformIAddOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformIAddOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformIMulOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformIMulOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformIMulOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
      attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getAttr<::mlir::spirv::GroupOperationAttr>(static_cast<::mlir::spirv::GroupOperation>(words[wordIndex++]))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformIMulOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformIMulOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformSMaxOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformSMaxOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformSMaxOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
      attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getAttr<::mlir::spirv::GroupOperationAttr>(static_cast<::mlir::spirv::GroupOperation>(words[wordIndex++]))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformSMaxOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformSMaxOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformSMinOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformSMinOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformSMinOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
      attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getAttr<::mlir::spirv::GroupOperationAttr>(static_cast<::mlir::spirv::GroupOperation>(words[wordIndex++]))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformSMinOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformSMinOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformShuffleDownOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformShuffleDownOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformShuffleDownOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformShuffleDownOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformShuffleDownOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformShuffleOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformShuffleOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformShuffleOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformShuffleOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformShuffleOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformShuffleUpOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformShuffleUpOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformShuffleUpOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformShuffleUpOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformShuffleUpOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformShuffleXorOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformShuffleXorOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformShuffleXorOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformShuffleXorOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformShuffleXorOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformUMaxOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformUMaxOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformUMaxOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
      attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getAttr<::mlir::spirv::GroupOperationAttr>(static_cast<::mlir::spirv::GroupOperation>(words[wordIndex++]))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformUMaxOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformUMaxOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformUMinOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformUMinOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformUMinOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
      attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getAttr<::mlir::spirv::GroupOperationAttr>(static_cast<::mlir::spirv::GroupOperation>(words[wordIndex++]))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformUMinOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformUMinOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::IAddCarryOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.IAddCarry", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::IAddOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.IAdd", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::IEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.IEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::IMulOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.IMul", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::INTELJointMatrixLoadOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::INTELJointMatrixLoadOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::INTELJointMatrixLoadOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("layout", opBuilder.getAttr<::mlir::spirv::MatrixLayoutAttr>(static_cast<::mlir::spirv::MatrixLayout>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
      attributes.push_back(opBuilder.getNamedAttr("memory_access", opBuilder.getAttr<::mlir::spirv::MemoryAccessAttr>(static_cast<::mlir::spirv::MemoryAccess>(words[wordIndex++]))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("alignment", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::INTELJointMatrixLoadOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::INTELJointMatrixLoadOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::INTELJointMatrixMadOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::INTELJointMatrixMadOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::INTELJointMatrixMadOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::INTELJointMatrixMadOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::INTELJointMatrixMadOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::INTELJointMatrixStoreOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("layout", opBuilder.getAttr<::mlir::spirv::MatrixLayoutAttr>(static_cast<::mlir::spirv::MatrixLayout>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
      attributes.push_back(opBuilder.getNamedAttr("memory_access", opBuilder.getAttr<::mlir::spirv::MemoryAccessAttr>(static_cast<::mlir::spirv::MemoryAccess>(words[wordIndex++]))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("alignment", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::INTELJointMatrixStoreOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::INTELJointMatrixStoreOp>(loc, resultTypes, operands, attributes); (void)op;
  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::INTELJointMatrixWorkItemLengthOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::INTELJointMatrixWorkItemLengthOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::INTELJointMatrixWorkItemLengthOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("type", TypeAttr::get(getType(words[wordIndex++]))));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::INTELJointMatrixWorkItemLengthOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::INTELJointMatrixWorkItemLengthOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::INTELSubgroupBlockReadOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.INTEL.SubgroupBlockRead", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::INTELSubgroupBlockWriteOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.INTEL.SubgroupBlockWrite", false, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::INotEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.INotEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ISubBorrowOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ISubBorrow", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ISubOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ISub", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ImageDrefGatherOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::ImageDrefGatherOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::ImageDrefGatherOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
      attributes.push_back(opBuilder.getNamedAttr("imageoperands", opBuilder.getAttr<::mlir::spirv::ImageOperandsAttr>(static_cast<::mlir::spirv::ImageOperands>(words[wordIndex++]))));
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::ImageDrefGatherOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::ImageDrefGatherOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ImageOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.Image", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ImageQuerySizeOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ImageQuerySize", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::InBoundsPtrAccessChainOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::InBoundsPtrAccessChainOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::InBoundsPtrAccessChainOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::InBoundsPtrAccessChainOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::InBoundsPtrAccessChainOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::IsInfOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.IsInf", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::IsNanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.IsNan", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::KHRAssumeTrueOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.KHR.AssumeTrue", false, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::KHRSubgroupBallotOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.KHR.SubgroupBallot", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::LoadOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::LoadOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::LoadOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
      attributes.push_back(opBuilder.getNamedAttr("memory_access", opBuilder.getAttr<::mlir::spirv::MemoryAccessAttr>(static_cast<::mlir::spirv::MemoryAccess>(words[wordIndex++]))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("alignment", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::LoadOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::LoadOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::LogicalAndOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.LogicalAnd", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::LogicalEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.LogicalEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::LogicalNotEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.LogicalNotEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::LogicalNotOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.LogicalNot", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::LogicalOrOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.LogicalOr", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::MatrixTimesMatrixOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.MatrixTimesMatrix", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::MatrixTimesScalarOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.MatrixTimesScalar", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::MemoryBarrierOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", opBuilder.getAttr<::mlir::spirv::ScopeAttr>(static_cast<::mlir::spirv::Scope>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_semantics", opBuilder.getAttr<::mlir::spirv::MemorySemanticsAttr>(static_cast<::mlir::spirv::MemorySemantics>(getConstantInt(words[wordIndex++]).getValue().getZExtValue()))));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::MemoryBarrierOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::MemoryBarrierOp>(loc, resultTypes, operands, attributes); (void)op;
  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::NVCooperativeMatrixLengthOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::NVCooperativeMatrixLengthOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::NVCooperativeMatrixLengthOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("type", TypeAttr::get(getType(words[wordIndex++]))));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::NVCooperativeMatrixLengthOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::NVCooperativeMatrixLengthOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::NVCooperativeMatrixLoadOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::NVCooperativeMatrixLoadOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::NVCooperativeMatrixLoadOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
      attributes.push_back(opBuilder.getNamedAttr("memory_access", opBuilder.getAttr<::mlir::spirv::MemoryAccessAttr>(static_cast<::mlir::spirv::MemoryAccess>(words[wordIndex++]))));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::NVCooperativeMatrixLoadOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::NVCooperativeMatrixLoadOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::NVCooperativeMatrixMulAddOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.NV.CooperativeMatrixMulAdd", true, 3);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::NVCooperativeMatrixStoreOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
      attributes.push_back(opBuilder.getNamedAttr("memory_access", opBuilder.getAttr<::mlir::spirv::MemoryAccessAttr>(static_cast<::mlir::spirv::MemoryAccess>(words[wordIndex++]))));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::NVCooperativeMatrixStoreOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::NVCooperativeMatrixStoreOp>(loc, resultTypes, operands, attributes); (void)op;
  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::NotOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.Not", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::OrderedOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.Ordered", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::PtrAccessChainOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::PtrAccessChainOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::PtrAccessChainOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::PtrAccessChainOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::PtrAccessChainOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ReturnOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.Return", false, 0);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ReturnValueOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ReturnValue", false, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SConvertOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SConvert", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SDivOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SDiv", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SGreaterThanEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SGreaterThanEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SGreaterThanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SGreaterThan", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SLessThanEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SLessThanEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SLessThanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SLessThan", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SModOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SMod", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SNegateOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SNegate", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SRemOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.SRem", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SelectOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.Select", true, 3);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ShiftLeftLogicalOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ShiftLeftLogical", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ShiftRightArithmeticOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ShiftRightArithmetic", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ShiftRightLogicalOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ShiftRightLogical", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::StoreOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
      attributes.push_back(opBuilder.getNamedAttr("memory_access", opBuilder.getAttr<::mlir::spirv::MemoryAccessAttr>(static_cast<::mlir::spirv::MemoryAccess>(words[wordIndex++]))));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("alignment", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::StoreOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::StoreOp>(loc, resultTypes, operands, attributes); (void)op;
  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::TransposeOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.Transpose", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::UConvertOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.UConvert", true, 1);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::UDivOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.UDiv", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::UGreaterThanEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.UGreaterThanEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::UGreaterThanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.UGreaterThan", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ULessThanEqualOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ULessThanEqual", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ULessThanOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.ULessThan", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::UModOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.UMod", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::UnorderedOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.Unordered", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::UnreachableOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.Unreachable", false, 0);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::VariableOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::VariableOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::VariableOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
      attributes.push_back(opBuilder.getNamedAttr("storage_class", opBuilder.getAttr<::mlir::spirv::StorageClassAttr>(static_cast<::mlir::spirv::StorageClass>(words[wordIndex++]))));
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::VariableOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::VariableOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::VectorExtractDynamicOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.VectorExtractDynamic", true, 2);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::VectorInsertDynamicOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.VectorInsertDynamic", true, 3);
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::VectorShuffleOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::VectorShuffleOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::VectorShuffleOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    SmallVector<Attribute, 4> attrListElems;
    while (wordIndex < words.size()) {
      attrListElems.push_back(opBuilder.getI32IntegerAttr(words[wordIndex++]));
    }
    attributes.push_back(opBuilder.getNamedAttr("components", opBuilder.getArrayAttr(attrListElems)));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::VectorShuffleOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::VectorShuffleOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    (void)clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::VectorTimesScalarOp>(ArrayRef<uint32_t> words) {
  return processOpWithoutGrammarAttr(words, "spv.VectorTimesScalar", true, 2);
}

LogicalResult spirv::Deserializer::dispatchToAutogenDeserialization(spirv::Opcode opcode, ArrayRef<uint32_t> words) {
  switch (opcode) {
  case spirv::Opcode::OpAccessChain:
    return processOp<::mlir::spirv::AccessChainOp>(words);
  case spirv::Opcode::OpAtomicAnd:
    return processOp<::mlir::spirv::AtomicAndOp>(words);
  case spirv::Opcode::OpAtomicCompareExchange:
    return processOp<::mlir::spirv::AtomicCompareExchangeOp>(words);
  case spirv::Opcode::OpAtomicCompareExchangeWeak:
    return processOp<::mlir::spirv::AtomicCompareExchangeWeakOp>(words);
  case spirv::Opcode::OpAtomicExchange:
    return processOp<::mlir::spirv::AtomicExchangeOp>(words);
  case spirv::Opcode::OpAtomicIAdd:
    return processOp<::mlir::spirv::AtomicIAddOp>(words);
  case spirv::Opcode::OpAtomicIDecrement:
    return processOp<::mlir::spirv::AtomicIDecrementOp>(words);
  case spirv::Opcode::OpAtomicIIncrement:
    return processOp<::mlir::spirv::AtomicIIncrementOp>(words);
  case spirv::Opcode::OpAtomicISub:
    return processOp<::mlir::spirv::AtomicISubOp>(words);
  case spirv::Opcode::OpAtomicOr:
    return processOp<::mlir::spirv::AtomicOrOp>(words);
  case spirv::Opcode::OpAtomicSMax:
    return processOp<::mlir::spirv::AtomicSMaxOp>(words);
  case spirv::Opcode::OpAtomicSMin:
    return processOp<::mlir::spirv::AtomicSMinOp>(words);
  case spirv::Opcode::OpAtomicUMax:
    return processOp<::mlir::spirv::AtomicUMaxOp>(words);
  case spirv::Opcode::OpAtomicUMin:
    return processOp<::mlir::spirv::AtomicUMinOp>(words);
  case spirv::Opcode::OpAtomicXor:
    return processOp<::mlir::spirv::AtomicXorOp>(words);
  case spirv::Opcode::OpBitCount:
    return processOp<::mlir::spirv::BitCountOp>(words);
  case spirv::Opcode::OpBitFieldInsert:
    return processOp<::mlir::spirv::BitFieldInsertOp>(words);
  case spirv::Opcode::OpBitFieldSExtract:
    return processOp<::mlir::spirv::BitFieldSExtractOp>(words);
  case spirv::Opcode::OpBitFieldUExtract:
    return processOp<::mlir::spirv::BitFieldUExtractOp>(words);
  case spirv::Opcode::OpBitReverse:
    return processOp<::mlir::spirv::BitReverseOp>(words);
  case spirv::Opcode::OpBitcast:
    return processOp<::mlir::spirv::BitcastOp>(words);
  case spirv::Opcode::OpBitwiseAnd:
    return processOp<::mlir::spirv::BitwiseAndOp>(words);
  case spirv::Opcode::OpBitwiseOr:
    return processOp<::mlir::spirv::BitwiseOrOp>(words);
  case spirv::Opcode::OpBitwiseXor:
    return processOp<::mlir::spirv::BitwiseXorOp>(words);
  case spirv::Opcode::OpBranchConditional:
    return processOp<::mlir::spirv::BranchConditionalOp>(words);
  case spirv::Opcode::OpBranch:
    return processOp<::mlir::spirv::BranchOp>(words);
  case spirv::Opcode::OpCompositeConstruct:
    return processOp<::mlir::spirv::CompositeConstructOp>(words);
  case spirv::Opcode::OpCompositeExtract:
    return processOp<::mlir::spirv::CompositeExtractOp>(words);
  case spirv::Opcode::OpCompositeInsert:
    return processOp<::mlir::spirv::CompositeInsertOp>(words);
  case spirv::Opcode::OpControlBarrier:
    return processOp<::mlir::spirv::ControlBarrierOp>(words);
  case spirv::Opcode::OpConvertFToS:
    return processOp<::mlir::spirv::ConvertFToSOp>(words);
  case spirv::Opcode::OpConvertFToU:
    return processOp<::mlir::spirv::ConvertFToUOp>(words);
  case spirv::Opcode::OpConvertSToF:
    return processOp<::mlir::spirv::ConvertSToFOp>(words);
  case spirv::Opcode::OpConvertUToF:
    return processOp<::mlir::spirv::ConvertUToFOp>(words);
  case spirv::Opcode::OpCopyMemory:
    return processOp<::mlir::spirv::CopyMemoryOp>(words);
  case spirv::Opcode::OpAtomicFAddEXT:
    return processOp<::mlir::spirv::EXTAtomicFAddOp>(words);
  case spirv::Opcode::OpEntryPoint:
    return processOp<::mlir::spirv::EntryPointOp>(words);
  case spirv::Opcode::OpExecutionMode:
    return processOp<::mlir::spirv::ExecutionModeOp>(words);
  case spirv::Opcode::OpFAdd:
    return processOp<::mlir::spirv::FAddOp>(words);
  case spirv::Opcode::OpFConvert:
    return processOp<::mlir::spirv::FConvertOp>(words);
  case spirv::Opcode::OpFDiv:
    return processOp<::mlir::spirv::FDivOp>(words);
  case spirv::Opcode::OpFMod:
    return processOp<::mlir::spirv::FModOp>(words);
  case spirv::Opcode::OpFMul:
    return processOp<::mlir::spirv::FMulOp>(words);
  case spirv::Opcode::OpFNegate:
    return processOp<::mlir::spirv::FNegateOp>(words);
  case spirv::Opcode::OpFOrdEqual:
    return processOp<::mlir::spirv::FOrdEqualOp>(words);
  case spirv::Opcode::OpFOrdGreaterThanEqual:
    return processOp<::mlir::spirv::FOrdGreaterThanEqualOp>(words);
  case spirv::Opcode::OpFOrdGreaterThan:
    return processOp<::mlir::spirv::FOrdGreaterThanOp>(words);
  case spirv::Opcode::OpFOrdLessThanEqual:
    return processOp<::mlir::spirv::FOrdLessThanEqualOp>(words);
  case spirv::Opcode::OpFOrdLessThan:
    return processOp<::mlir::spirv::FOrdLessThanOp>(words);
  case spirv::Opcode::OpFOrdNotEqual:
    return processOp<::mlir::spirv::FOrdNotEqualOp>(words);
  case spirv::Opcode::OpFRem:
    return processOp<::mlir::spirv::FRemOp>(words);
  case spirv::Opcode::OpFSub:
    return processOp<::mlir::spirv::FSubOp>(words);
  case spirv::Opcode::OpFUnordEqual:
    return processOp<::mlir::spirv::FUnordEqualOp>(words);
  case spirv::Opcode::OpFUnordGreaterThanEqual:
    return processOp<::mlir::spirv::FUnordGreaterThanEqualOp>(words);
  case spirv::Opcode::OpFUnordGreaterThan:
    return processOp<::mlir::spirv::FUnordGreaterThanOp>(words);
  case spirv::Opcode::OpFUnordLessThanEqual:
    return processOp<::mlir::spirv::FUnordLessThanEqualOp>(words);
  case spirv::Opcode::OpFUnordLessThan:
    return processOp<::mlir::spirv::FUnordLessThanOp>(words);
  case spirv::Opcode::OpFUnordNotEqual:
    return processOp<::mlir::spirv::FUnordNotEqualOp>(words);
  case spirv::Opcode::OpFunctionCall:
    return processOp<::mlir::spirv::FunctionCallOp>(words);
  case spirv::Opcode::OpGroupBroadcast:
    return processOp<::mlir::spirv::GroupBroadcastOp>(words);
  case spirv::Opcode::OpGroupNonUniformBallot:
    return processOp<::mlir::spirv::GroupNonUniformBallotOp>(words);
  case spirv::Opcode::OpGroupNonUniformBroadcast:
    return processOp<::mlir::spirv::GroupNonUniformBroadcastOp>(words);
  case spirv::Opcode::OpGroupNonUniformElect:
    return processOp<::mlir::spirv::GroupNonUniformElectOp>(words);
  case spirv::Opcode::OpGroupNonUniformFAdd:
    return processOp<::mlir::spirv::GroupNonUniformFAddOp>(words);
  case spirv::Opcode::OpGroupNonUniformFMax:
    return processOp<::mlir::spirv::GroupNonUniformFMaxOp>(words);
  case spirv::Opcode::OpGroupNonUniformFMin:
    return processOp<::mlir::spirv::GroupNonUniformFMinOp>(words);
  case spirv::Opcode::OpGroupNonUniformFMul:
    return processOp<::mlir::spirv::GroupNonUniformFMulOp>(words);
  case spirv::Opcode::OpGroupNonUniformIAdd:
    return processOp<::mlir::spirv::GroupNonUniformIAddOp>(words);
  case spirv::Opcode::OpGroupNonUniformIMul:
    return processOp<::mlir::spirv::GroupNonUniformIMulOp>(words);
  case spirv::Opcode::OpGroupNonUniformSMax:
    return processOp<::mlir::spirv::GroupNonUniformSMaxOp>(words);
  case spirv::Opcode::OpGroupNonUniformSMin:
    return processOp<::mlir::spirv::GroupNonUniformSMinOp>(words);
  case spirv::Opcode::OpGroupNonUniformShuffleDown:
    return processOp<::mlir::spirv::GroupNonUniformShuffleDownOp>(words);
  case spirv::Opcode::OpGroupNonUniformShuffle:
    return processOp<::mlir::spirv::GroupNonUniformShuffleOp>(words);
  case spirv::Opcode::OpGroupNonUniformShuffleUp:
    return processOp<::mlir::spirv::GroupNonUniformShuffleUpOp>(words);
  case spirv::Opcode::OpGroupNonUniformShuffleXor:
    return processOp<::mlir::spirv::GroupNonUniformShuffleXorOp>(words);
  case spirv::Opcode::OpGroupNonUniformUMax:
    return processOp<::mlir::spirv::GroupNonUniformUMaxOp>(words);
  case spirv::Opcode::OpGroupNonUniformUMin:
    return processOp<::mlir::spirv::GroupNonUniformUMinOp>(words);
  case spirv::Opcode::OpIAddCarry:
    return processOp<::mlir::spirv::IAddCarryOp>(words);
  case spirv::Opcode::OpIAdd:
    return processOp<::mlir::spirv::IAddOp>(words);
  case spirv::Opcode::OpIEqual:
    return processOp<::mlir::spirv::IEqualOp>(words);
  case spirv::Opcode::OpIMul:
    return processOp<::mlir::spirv::IMulOp>(words);
  case spirv::Opcode::OpJointMatrixLoadINTEL:
    return processOp<::mlir::spirv::INTELJointMatrixLoadOp>(words);
  case spirv::Opcode::OpJointMatrixMadINTEL:
    return processOp<::mlir::spirv::INTELJointMatrixMadOp>(words);
  case spirv::Opcode::OpJointMatrixStoreINTEL:
    return processOp<::mlir::spirv::INTELJointMatrixStoreOp>(words);
  case spirv::Opcode::OpJointMatrixWorkItemLengthINTEL:
    return processOp<::mlir::spirv::INTELJointMatrixWorkItemLengthOp>(words);
  case spirv::Opcode::OpSubgroupBlockReadINTEL:
    return processOp<::mlir::spirv::INTELSubgroupBlockReadOp>(words);
  case spirv::Opcode::OpSubgroupBlockWriteINTEL:
    return processOp<::mlir::spirv::INTELSubgroupBlockWriteOp>(words);
  case spirv::Opcode::OpINotEqual:
    return processOp<::mlir::spirv::INotEqualOp>(words);
  case spirv::Opcode::OpISubBorrow:
    return processOp<::mlir::spirv::ISubBorrowOp>(words);
  case spirv::Opcode::OpISub:
    return processOp<::mlir::spirv::ISubOp>(words);
  case spirv::Opcode::OpImageDrefGather:
    return processOp<::mlir::spirv::ImageDrefGatherOp>(words);
  case spirv::Opcode::OpImage:
    return processOp<::mlir::spirv::ImageOp>(words);
  case spirv::Opcode::OpImageQuerySize:
    return processOp<::mlir::spirv::ImageQuerySizeOp>(words);
  case spirv::Opcode::OpInBoundsPtrAccessChain:
    return processOp<::mlir::spirv::InBoundsPtrAccessChainOp>(words);
  case spirv::Opcode::OpIsInf:
    return processOp<::mlir::spirv::IsInfOp>(words);
  case spirv::Opcode::OpIsNan:
    return processOp<::mlir::spirv::IsNanOp>(words);
  case spirv::Opcode::OpAssumeTrueKHR:
    return processOp<::mlir::spirv::KHRAssumeTrueOp>(words);
  case spirv::Opcode::OpSubgroupBallotKHR:
    return processOp<::mlir::spirv::KHRSubgroupBallotOp>(words);
  case spirv::Opcode::OpLoad:
    return processOp<::mlir::spirv::LoadOp>(words);
  case spirv::Opcode::OpLogicalAnd:
    return processOp<::mlir::spirv::LogicalAndOp>(words);
  case spirv::Opcode::OpLogicalEqual:
    return processOp<::mlir::spirv::LogicalEqualOp>(words);
  case spirv::Opcode::OpLogicalNotEqual:
    return processOp<::mlir::spirv::LogicalNotEqualOp>(words);
  case spirv::Opcode::OpLogicalNot:
    return processOp<::mlir::spirv::LogicalNotOp>(words);
  case spirv::Opcode::OpLogicalOr:
    return processOp<::mlir::spirv::LogicalOrOp>(words);
  case spirv::Opcode::OpMatrixTimesMatrix:
    return processOp<::mlir::spirv::MatrixTimesMatrixOp>(words);
  case spirv::Opcode::OpMatrixTimesScalar:
    return processOp<::mlir::spirv::MatrixTimesScalarOp>(words);
  case spirv::Opcode::OpMemoryBarrier:
    return processOp<::mlir::spirv::MemoryBarrierOp>(words);
  case spirv::Opcode::OpCooperativeMatrixLengthNV:
    return processOp<::mlir::spirv::NVCooperativeMatrixLengthOp>(words);
  case spirv::Opcode::OpCooperativeMatrixLoadNV:
    return processOp<::mlir::spirv::NVCooperativeMatrixLoadOp>(words);
  case spirv::Opcode::OpCooperativeMatrixMulAddNV:
    return processOp<::mlir::spirv::NVCooperativeMatrixMulAddOp>(words);
  case spirv::Opcode::OpCooperativeMatrixStoreNV:
    return processOp<::mlir::spirv::NVCooperativeMatrixStoreOp>(words);
  case spirv::Opcode::OpNot:
    return processOp<::mlir::spirv::NotOp>(words);
  case spirv::Opcode::OpOrdered:
    return processOp<::mlir::spirv::OrderedOp>(words);
  case spirv::Opcode::OpPtrAccessChain:
    return processOp<::mlir::spirv::PtrAccessChainOp>(words);
  case spirv::Opcode::OpReturn:
    return processOp<::mlir::spirv::ReturnOp>(words);
  case spirv::Opcode::OpReturnValue:
    return processOp<::mlir::spirv::ReturnValueOp>(words);
  case spirv::Opcode::OpSConvert:
    return processOp<::mlir::spirv::SConvertOp>(words);
  case spirv::Opcode::OpSDiv:
    return processOp<::mlir::spirv::SDivOp>(words);
  case spirv::Opcode::OpSGreaterThanEqual:
    return processOp<::mlir::spirv::SGreaterThanEqualOp>(words);
  case spirv::Opcode::OpSGreaterThan:
    return processOp<::mlir::spirv::SGreaterThanOp>(words);
  case spirv::Opcode::OpSLessThanEqual:
    return processOp<::mlir::spirv::SLessThanEqualOp>(words);
  case spirv::Opcode::OpSLessThan:
    return processOp<::mlir::spirv::SLessThanOp>(words);
  case spirv::Opcode::OpSMod:
    return processOp<::mlir::spirv::SModOp>(words);
  case spirv::Opcode::OpSNegate:
    return processOp<::mlir::spirv::SNegateOp>(words);
  case spirv::Opcode::OpSRem:
    return processOp<::mlir::spirv::SRemOp>(words);
  case spirv::Opcode::OpSelect:
    return processOp<::mlir::spirv::SelectOp>(words);
  case spirv::Opcode::OpShiftLeftLogical:
    return processOp<::mlir::spirv::ShiftLeftLogicalOp>(words);
  case spirv::Opcode::OpShiftRightArithmetic:
    return processOp<::mlir::spirv::ShiftRightArithmeticOp>(words);
  case spirv::Opcode::OpShiftRightLogical:
    return processOp<::mlir::spirv::ShiftRightLogicalOp>(words);
  case spirv::Opcode::OpStore:
    return processOp<::mlir::spirv::StoreOp>(words);
  case spirv::Opcode::OpTranspose:
    return processOp<::mlir::spirv::TransposeOp>(words);
  case spirv::Opcode::OpUConvert:
    return processOp<::mlir::spirv::UConvertOp>(words);
  case spirv::Opcode::OpUDiv:
    return processOp<::mlir::spirv::UDivOp>(words);
  case spirv::Opcode::OpUGreaterThanEqual:
    return processOp<::mlir::spirv::UGreaterThanEqualOp>(words);
  case spirv::Opcode::OpUGreaterThan:
    return processOp<::mlir::spirv::UGreaterThanOp>(words);
  case spirv::Opcode::OpULessThanEqual:
    return processOp<::mlir::spirv::ULessThanEqualOp>(words);
  case spirv::Opcode::OpULessThan:
    return processOp<::mlir::spirv::ULessThanOp>(words);
  case spirv::Opcode::OpUMod:
    return processOp<::mlir::spirv::UModOp>(words);
  case spirv::Opcode::OpUnordered:
    return processOp<::mlir::spirv::UnorderedOp>(words);
  case spirv::Opcode::OpUnreachable:
    return processOp<::mlir::spirv::UnreachableOp>(words);
  case spirv::Opcode::OpVariable:
    return processOp<::mlir::spirv::VariableOp>(words);
  case spirv::Opcode::OpVectorExtractDynamic:
    return processOp<::mlir::spirv::VectorExtractDynamicOp>(words);
  case spirv::Opcode::OpVectorInsertDynamic:
    return processOp<::mlir::spirv::VectorInsertDynamicOp>(words);
  case spirv::Opcode::OpVectorShuffle:
    return processOp<::mlir::spirv::VectorShuffleOp>(words);
  case spirv::Opcode::OpVectorTimesScalar:
    return processOp<::mlir::spirv::VectorTimesScalarOp>(words);
  default:
    ;
  }
  auto opcodeString = spirv::stringifyOpcode(opcode);
  if (!opcodeString.empty()) {
    return emitError(unknownLoc, "unhandled deserialization of ") << opcodeString;
  } else {
   return emitError(unknownLoc, "unhandled opcode ") << static_cast<uint32_t>(opcode);
  }
}
LogicalResult spirv::Deserializer::dispatchToExtensionSetAutogenDeserialization(StringRef extensionSetName, uint32_t instructionID, ArrayRef<uint32_t> words) {
  if (extensionSetName == "GLSL.std.450") {
    switch (instructionID) {
    case 17:
      return processOp<::mlir::spirv::GLAcosOp>(words);
    case 16:
      return processOp<::mlir::spirv::GLAsinOp>(words);
    case 18:
      return processOp<::mlir::spirv::GLAtanOp>(words);
    case 9:
      return processOp<::mlir::spirv::GLCeilOp>(words);
    case 14:
      return processOp<::mlir::spirv::GLCosOp>(words);
    case 20:
      return processOp<::mlir::spirv::GLCoshOp>(words);
    case 27:
      return processOp<::mlir::spirv::GLExpOp>(words);
    case 4:
      return processOp<::mlir::spirv::GLFAbsOp>(words);
    case 43:
      return processOp<::mlir::spirv::GLFClampOp>(words);
    case 40:
      return processOp<::mlir::spirv::GLFMaxOp>(words);
    case 37:
      return processOp<::mlir::spirv::GLFMinOp>(words);
    case 46:
      return processOp<::mlir::spirv::GLFMixOp>(words);
    case 6:
      return processOp<::mlir::spirv::GLFSignOp>(words);
    case 75:
      return processOp<::mlir::spirv::GLFindUMsbOp>(words);
    case 8:
      return processOp<::mlir::spirv::GLFloorOp>(words);
    case 50:
      return processOp<::mlir::spirv::GLFmaOp>(words);
    case 52:
      return processOp<::mlir::spirv::GLFrexpStructOp>(words);
    case 32:
      return processOp<::mlir::spirv::GLInverseSqrtOp>(words);
    case 53:
      return processOp<::mlir::spirv::GLLdexpOp>(words);
    case 28:
      return processOp<::mlir::spirv::GLLogOp>(words);
    case 26:
      return processOp<::mlir::spirv::GLPowOp>(words);
    case 1:
      return processOp<::mlir::spirv::GLRoundOp>(words);
    case 5:
      return processOp<::mlir::spirv::GLSAbsOp>(words);
    case 45:
      return processOp<::mlir::spirv::GLSClampOp>(words);
    case 42:
      return processOp<::mlir::spirv::GLSMaxOp>(words);
    case 39:
      return processOp<::mlir::spirv::GLSMinOp>(words);
    case 7:
      return processOp<::mlir::spirv::GLSSignOp>(words);
    case 13:
      return processOp<::mlir::spirv::GLSinOp>(words);
    case 19:
      return processOp<::mlir::spirv::GLSinhOp>(words);
    case 31:
      return processOp<::mlir::spirv::GLSqrtOp>(words);
    case 15:
      return processOp<::mlir::spirv::GLTanOp>(words);
    case 21:
      return processOp<::mlir::spirv::GLTanhOp>(words);
    case 44:
      return processOp<::mlir::spirv::GLUClampOp>(words);
    case 41:
      return processOp<::mlir::spirv::GLUMaxOp>(words);
    case 38:
      return processOp<::mlir::spirv::GLUMinOp>(words);
    default:
      return emitError(unknownLoc, "unhandled deserializations of ") << instructionID << " from extension set " << extensionSetName;
    }
  }
  if (extensionSetName == "OpenCL.std") {
    switch (instructionID) {
    case 12:
      return processOp<::mlir::spirv::CLCeilOp>(words);
    case 14:
      return processOp<::mlir::spirv::CLCosOp>(words);
    case 18:
      return processOp<::mlir::spirv::CLErfOp>(words);
    case 19:
      return processOp<::mlir::spirv::CLExpOp>(words);
    case 23:
      return processOp<::mlir::spirv::CLFAbsOp>(words);
    case 25:
      return processOp<::mlir::spirv::CLFloorOp>(words);
    case 26:
      return processOp<::mlir::spirv::CLFmaOp>(words);
    case 37:
      return processOp<::mlir::spirv::CLLogOp>(words);
    case 48:
      return processOp<::mlir::spirv::CLPowOp>(words);
    case 55:
      return processOp<::mlir::spirv::CLRoundOp>(words);
    case 56:
      return processOp<::mlir::spirv::CLRsqrtOp>(words);
    case 141:
      return processOp<::mlir::spirv::CLSAbsOp>(words);
    case 57:
      return processOp<::mlir::spirv::CLSinOp>(words);
    case 61:
      return processOp<::mlir::spirv::CLSqrtOp>(words);
    case 63:
      return processOp<::mlir::spirv::CLTanhOp>(words);
    default:
      return emitError(unknownLoc, "unhandled deserializations of ") << instructionID << " from extension set " << extensionSetName;
    }
  }
  return emitError(unknownLoc, "unhandled deserialization of extended instruction set extensionSetName");
}
#endif // GET_DESERIALIZATION_FNS

