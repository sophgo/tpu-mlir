static LLVM_ATTRIBUTE_UNUSED ::llvm::InlineAsm::AsmDialect convertAsmDialectToLLVM(::mlir::LLVM::AsmDialect value) {
  switch (value) {
  case ::mlir::LLVM::AsmDialect::AD_ATT:
    return ::llvm::InlineAsm::AsmDialect::AD_ATT;
  case ::mlir::LLVM::AsmDialect::AD_Intel:
    return ::llvm::InlineAsm::AsmDialect::AD_Intel;
  }
  llvm_unreachable("unknown AsmDialect type");
}

static LLVM_ATTRIBUTE_UNUSED ::llvm::GlobalValue::LinkageTypes convertLinkageToLLVM(::mlir::LLVM::linkage::Linkage value) {
  switch (value) {
  case ::mlir::LLVM::linkage::Linkage::Private:
    return ::llvm::GlobalValue::LinkageTypes::PrivateLinkage;
  case ::mlir::LLVM::linkage::Linkage::Internal:
    return ::llvm::GlobalValue::LinkageTypes::InternalLinkage;
  case ::mlir::LLVM::linkage::Linkage::AvailableExternally:
    return ::llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage;
  case ::mlir::LLVM::linkage::Linkage::Linkonce:
    return ::llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage;
  case ::mlir::LLVM::linkage::Linkage::Weak:
    return ::llvm::GlobalValue::LinkageTypes::WeakAnyLinkage;
  case ::mlir::LLVM::linkage::Linkage::Common:
    return ::llvm::GlobalValue::LinkageTypes::CommonLinkage;
  case ::mlir::LLVM::linkage::Linkage::Appending:
    return ::llvm::GlobalValue::LinkageTypes::AppendingLinkage;
  case ::mlir::LLVM::linkage::Linkage::ExternWeak:
    return ::llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage;
  case ::mlir::LLVM::linkage::Linkage::LinkonceODR:
    return ::llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage;
  case ::mlir::LLVM::linkage::Linkage::WeakODR:
    return ::llvm::GlobalValue::LinkageTypes::WeakODRLinkage;
  case ::mlir::LLVM::linkage::Linkage::External:
    return ::llvm::GlobalValue::LinkageTypes::ExternalLinkage;
  }
  llvm_unreachable("unknown Linkage type");
}

static LLVM_ATTRIBUTE_UNUSED ::llvm::GlobalValue::UnnamedAddr convertUnnamedAddrToLLVM(::mlir::LLVM::UnnamedAddr value) {
  switch (value) {
  case ::mlir::LLVM::UnnamedAddr::None:
    return ::llvm::GlobalValue::UnnamedAddr::None;
  case ::mlir::LLVM::UnnamedAddr::Local:
    return ::llvm::GlobalValue::UnnamedAddr::Local;
  case ::mlir::LLVM::UnnamedAddr::Global:
    return ::llvm::GlobalValue::UnnamedAddr::Global;
  }
  llvm_unreachable("unknown UnnamedAddr type");
}

static LLVM_ATTRIBUTE_UNUSED int64_t convertCConvToLLVM(::mlir::LLVM::cconv::CConv value) {
  switch (value) {
  case ::mlir::LLVM::cconv::CConv::C:
    return static_cast<int64_t>(::llvm::CallingConv::C);
  case ::mlir::LLVM::cconv::CConv::Fast:
    return static_cast<int64_t>(::llvm::CallingConv::Fast);
  case ::mlir::LLVM::cconv::CConv::Cold:
    return static_cast<int64_t>(::llvm::CallingConv::Cold);
  case ::mlir::LLVM::cconv::CConv::GHC:
    return static_cast<int64_t>(::llvm::CallingConv::GHC);
  case ::mlir::LLVM::cconv::CConv::HiPE:
    return static_cast<int64_t>(::llvm::CallingConv::HiPE);
  case ::mlir::LLVM::cconv::CConv::WebKit_JS:
    return static_cast<int64_t>(::llvm::CallingConv::WebKit_JS);
  case ::mlir::LLVM::cconv::CConv::AnyReg:
    return static_cast<int64_t>(::llvm::CallingConv::AnyReg);
  case ::mlir::LLVM::cconv::CConv::PreserveMost:
    return static_cast<int64_t>(::llvm::CallingConv::PreserveMost);
  case ::mlir::LLVM::cconv::CConv::PreserveAll:
    return static_cast<int64_t>(::llvm::CallingConv::PreserveAll);
  case ::mlir::LLVM::cconv::CConv::Swift:
    return static_cast<int64_t>(::llvm::CallingConv::Swift);
  case ::mlir::LLVM::cconv::CConv::CXX_FAST_TLS:
    return static_cast<int64_t>(::llvm::CallingConv::CXX_FAST_TLS);
  case ::mlir::LLVM::cconv::CConv::Tail:
    return static_cast<int64_t>(::llvm::CallingConv::Tail);
  case ::mlir::LLVM::cconv::CConv::CFGuard_Check:
    return static_cast<int64_t>(::llvm::CallingConv::CFGuard_Check);
  case ::mlir::LLVM::cconv::CConv::SwiftTail:
    return static_cast<int64_t>(::llvm::CallingConv::SwiftTail);
  case ::mlir::LLVM::cconv::CConv::X86_StdCall:
    return static_cast<int64_t>(::llvm::CallingConv::X86_StdCall);
  case ::mlir::LLVM::cconv::CConv::X86_FastCall:
    return static_cast<int64_t>(::llvm::CallingConv::X86_FastCall);
  case ::mlir::LLVM::cconv::CConv::ARM_APCS:
    return static_cast<int64_t>(::llvm::CallingConv::ARM_APCS);
  case ::mlir::LLVM::cconv::CConv::ARM_AAPCS:
    return static_cast<int64_t>(::llvm::CallingConv::ARM_AAPCS);
  case ::mlir::LLVM::cconv::CConv::ARM_AAPCS_VFP:
    return static_cast<int64_t>(::llvm::CallingConv::ARM_AAPCS_VFP);
  case ::mlir::LLVM::cconv::CConv::MSP430_INTR:
    return static_cast<int64_t>(::llvm::CallingConv::MSP430_INTR);
  case ::mlir::LLVM::cconv::CConv::X86_ThisCall:
    return static_cast<int64_t>(::llvm::CallingConv::X86_ThisCall);
  case ::mlir::LLVM::cconv::CConv::PTX_Kernel:
    return static_cast<int64_t>(::llvm::CallingConv::PTX_Kernel);
  case ::mlir::LLVM::cconv::CConv::PTX_Device:
    return static_cast<int64_t>(::llvm::CallingConv::PTX_Device);
  case ::mlir::LLVM::cconv::CConv::SPIR_FUNC:
    return static_cast<int64_t>(::llvm::CallingConv::SPIR_FUNC);
  case ::mlir::LLVM::cconv::CConv::SPIR_KERNEL:
    return static_cast<int64_t>(::llvm::CallingConv::SPIR_KERNEL);
  case ::mlir::LLVM::cconv::CConv::Intel_OCL_BI:
    return static_cast<int64_t>(::llvm::CallingConv::Intel_OCL_BI);
  case ::mlir::LLVM::cconv::CConv::X86_64_SysV:
    return static_cast<int64_t>(::llvm::CallingConv::X86_64_SysV);
  case ::mlir::LLVM::cconv::CConv::Win64:
    return static_cast<int64_t>(::llvm::CallingConv::Win64);
  case ::mlir::LLVM::cconv::CConv::X86_VectorCall:
    return static_cast<int64_t>(::llvm::CallingConv::X86_VectorCall);
  case ::mlir::LLVM::cconv::CConv::HHVM:
    return static_cast<int64_t>(::llvm::CallingConv::HHVM);
  case ::mlir::LLVM::cconv::CConv::HHVM_C:
    return static_cast<int64_t>(::llvm::CallingConv::HHVM_C);
  case ::mlir::LLVM::cconv::CConv::X86_INTR:
    return static_cast<int64_t>(::llvm::CallingConv::X86_INTR);
  case ::mlir::LLVM::cconv::CConv::AVR_INTR:
    return static_cast<int64_t>(::llvm::CallingConv::AVR_INTR);
  case ::mlir::LLVM::cconv::CConv::AVR_BUILTIN:
    return static_cast<int64_t>(::llvm::CallingConv::AVR_BUILTIN);
  case ::mlir::LLVM::cconv::CConv::AMDGPU_VS:
    return static_cast<int64_t>(::llvm::CallingConv::AMDGPU_VS);
  case ::mlir::LLVM::cconv::CConv::AMDGPU_GS:
    return static_cast<int64_t>(::llvm::CallingConv::AMDGPU_GS);
  case ::mlir::LLVM::cconv::CConv::AMDGPU_CS:
    return static_cast<int64_t>(::llvm::CallingConv::AMDGPU_CS);
  case ::mlir::LLVM::cconv::CConv::AMDGPU_KERNEL:
    return static_cast<int64_t>(::llvm::CallingConv::AMDGPU_KERNEL);
  case ::mlir::LLVM::cconv::CConv::X86_RegCall:
    return static_cast<int64_t>(::llvm::CallingConv::X86_RegCall);
  case ::mlir::LLVM::cconv::CConv::AMDGPU_HS:
    return static_cast<int64_t>(::llvm::CallingConv::AMDGPU_HS);
  case ::mlir::LLVM::cconv::CConv::MSP430_BUILTIN:
    return static_cast<int64_t>(::llvm::CallingConv::MSP430_BUILTIN);
  case ::mlir::LLVM::cconv::CConv::AMDGPU_LS:
    return static_cast<int64_t>(::llvm::CallingConv::AMDGPU_LS);
  case ::mlir::LLVM::cconv::CConv::AMDGPU_ES:
    return static_cast<int64_t>(::llvm::CallingConv::AMDGPU_ES);
  case ::mlir::LLVM::cconv::CConv::AArch64_VectorCall:
    return static_cast<int64_t>(::llvm::CallingConv::AArch64_VectorCall);
  case ::mlir::LLVM::cconv::CConv::AArch64_SVE_VectorCall:
    return static_cast<int64_t>(::llvm::CallingConv::AArch64_SVE_VectorCall);
  case ::mlir::LLVM::cconv::CConv::WASM_EmscriptenInvoke:
    return static_cast<int64_t>(::llvm::CallingConv::WASM_EmscriptenInvoke);
  case ::mlir::LLVM::cconv::CConv::AMDGPU_Gfx:
    return static_cast<int64_t>(::llvm::CallingConv::AMDGPU_Gfx);
  case ::mlir::LLVM::cconv::CConv::M68k_INTR:
    return static_cast<int64_t>(::llvm::CallingConv::M68k_INTR);
  }
  llvm_unreachable("unknown CConv type");
}

