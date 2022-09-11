if (auto op = dyn_cast<::mlir::arm_neon::SMullOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::aarch64_neon_smull,
        { moduleTranslation.convertType(opInst.getResult(0).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::arm_neon::SdotOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::aarch64_neon_sdot,
        { moduleTranslation.convertType(opInst.getResult(0).getType()), moduleTranslation.convertType(opInst.getOperand(1).getType())
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
