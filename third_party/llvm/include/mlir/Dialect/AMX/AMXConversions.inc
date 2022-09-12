if (auto op = dyn_cast<::mlir::amx::x86_amx_tdpbf16ps>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::x86_tdpbf16ps_internal,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::amx::x86_amx_tdpbssd>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::x86_tdpbssd_internal,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::amx::x86_amx_tdpbsud>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::x86_tdpbsud_internal,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::amx::x86_amx_tdpbusd>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::x86_tdpbusd_internal,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::amx::x86_amx_tdpbuud>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::x86_tdpbuud_internal,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::amx::x86_amx_tileloadd64>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::x86_tileloadd64_internal,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::amx::x86_amx_tilestored64>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::x86_tilestored64_internal,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;
  return success();
}
if (auto op = dyn_cast<::mlir::amx::x86_amx_tilezero>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::x86_tilezero_internal,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
