inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::AsmDialect convertAsmDialectFromLLVM(::llvm::InlineAsm::AsmDialect value) {
  switch (value) {
  case ::llvm::InlineAsm::AsmDialect::AD_ATT:
    return ::mlir::LLVM::AsmDialect::AD_ATT;
  case ::llvm::InlineAsm::AsmDialect::AD_Intel:
    return ::mlir::LLVM::AsmDialect::AD_Intel;
  }
  llvm_unreachable("unknown ::llvm::InlineAsm::AsmDialect type");}

inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::linkage::Linkage convertLinkageFromLLVM(::llvm::GlobalValue::LinkageTypes value) {
  switch (value) {
  case ::llvm::GlobalValue::LinkageTypes::PrivateLinkage:
    return ::mlir::LLVM::linkage::Linkage::Private;
  case ::llvm::GlobalValue::LinkageTypes::InternalLinkage:
    return ::mlir::LLVM::linkage::Linkage::Internal;
  case ::llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage:
    return ::mlir::LLVM::linkage::Linkage::AvailableExternally;
  case ::llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage:
    return ::mlir::LLVM::linkage::Linkage::Linkonce;
  case ::llvm::GlobalValue::LinkageTypes::WeakAnyLinkage:
    return ::mlir::LLVM::linkage::Linkage::Weak;
  case ::llvm::GlobalValue::LinkageTypes::CommonLinkage:
    return ::mlir::LLVM::linkage::Linkage::Common;
  case ::llvm::GlobalValue::LinkageTypes::AppendingLinkage:
    return ::mlir::LLVM::linkage::Linkage::Appending;
  case ::llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage:
    return ::mlir::LLVM::linkage::Linkage::ExternWeak;
  case ::llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage:
    return ::mlir::LLVM::linkage::Linkage::LinkonceODR;
  case ::llvm::GlobalValue::LinkageTypes::WeakODRLinkage:
    return ::mlir::LLVM::linkage::Linkage::WeakODR;
  case ::llvm::GlobalValue::LinkageTypes::ExternalLinkage:
    return ::mlir::LLVM::linkage::Linkage::External;
  }
  llvm_unreachable("unknown ::llvm::GlobalValue::LinkageTypes type");}

inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::UnnamedAddr convertUnnamedAddrFromLLVM(::llvm::GlobalValue::UnnamedAddr value) {
  switch (value) {
  case ::llvm::GlobalValue::UnnamedAddr::None:
    return ::mlir::LLVM::UnnamedAddr::None;
  case ::llvm::GlobalValue::UnnamedAddr::Local:
    return ::mlir::LLVM::UnnamedAddr::Local;
  case ::llvm::GlobalValue::UnnamedAddr::Global:
    return ::mlir::LLVM::UnnamedAddr::Global;
  }
  llvm_unreachable("unknown ::llvm::GlobalValue::UnnamedAddr type");}

inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::cconv::CConv convertCConvFromLLVM(int64_t value) {
  switch (value) {
  case static_cast<int64_t>(::llvm::CallingConv::C):
    return ::mlir::LLVM::cconv::CConv::C;
  case static_cast<int64_t>(::llvm::CallingConv::Fast):
    return ::mlir::LLVM::cconv::CConv::Fast;
  case static_cast<int64_t>(::llvm::CallingConv::Cold):
    return ::mlir::LLVM::cconv::CConv::Cold;
  case static_cast<int64_t>(::llvm::CallingConv::GHC):
    return ::mlir::LLVM::cconv::CConv::GHC;
  case static_cast<int64_t>(::llvm::CallingConv::HiPE):
    return ::mlir::LLVM::cconv::CConv::HiPE;
  case static_cast<int64_t>(::llvm::CallingConv::WebKit_JS):
    return ::mlir::LLVM::cconv::CConv::WebKit_JS;
  case static_cast<int64_t>(::llvm::CallingConv::AnyReg):
    return ::mlir::LLVM::cconv::CConv::AnyReg;
  case static_cast<int64_t>(::llvm::CallingConv::PreserveMost):
    return ::mlir::LLVM::cconv::CConv::PreserveMost;
  case static_cast<int64_t>(::llvm::CallingConv::PreserveAll):
    return ::mlir::LLVM::cconv::CConv::PreserveAll;
  case static_cast<int64_t>(::llvm::CallingConv::Swift):
    return ::mlir::LLVM::cconv::CConv::Swift;
  case static_cast<int64_t>(::llvm::CallingConv::CXX_FAST_TLS):
    return ::mlir::LLVM::cconv::CConv::CXX_FAST_TLS;
  case static_cast<int64_t>(::llvm::CallingConv::Tail):
    return ::mlir::LLVM::cconv::CConv::Tail;
  case static_cast<int64_t>(::llvm::CallingConv::CFGuard_Check):
    return ::mlir::LLVM::cconv::CConv::CFGuard_Check;
  case static_cast<int64_t>(::llvm::CallingConv::SwiftTail):
    return ::mlir::LLVM::cconv::CConv::SwiftTail;
  case static_cast<int64_t>(::llvm::CallingConv::X86_StdCall):
    return ::mlir::LLVM::cconv::CConv::X86_StdCall;
  case static_cast<int64_t>(::llvm::CallingConv::X86_FastCall):
    return ::mlir::LLVM::cconv::CConv::X86_FastCall;
  case static_cast<int64_t>(::llvm::CallingConv::ARM_APCS):
    return ::mlir::LLVM::cconv::CConv::ARM_APCS;
  case static_cast<int64_t>(::llvm::CallingConv::ARM_AAPCS):
    return ::mlir::LLVM::cconv::CConv::ARM_AAPCS;
  case static_cast<int64_t>(::llvm::CallingConv::ARM_AAPCS_VFP):
    return ::mlir::LLVM::cconv::CConv::ARM_AAPCS_VFP;
  case static_cast<int64_t>(::llvm::CallingConv::MSP430_INTR):
    return ::mlir::LLVM::cconv::CConv::MSP430_INTR;
  case static_cast<int64_t>(::llvm::CallingConv::X86_ThisCall):
    return ::mlir::LLVM::cconv::CConv::X86_ThisCall;
  case static_cast<int64_t>(::llvm::CallingConv::PTX_Kernel):
    return ::mlir::LLVM::cconv::CConv::PTX_Kernel;
  case static_cast<int64_t>(::llvm::CallingConv::PTX_Device):
    return ::mlir::LLVM::cconv::CConv::PTX_Device;
  case static_cast<int64_t>(::llvm::CallingConv::SPIR_FUNC):
    return ::mlir::LLVM::cconv::CConv::SPIR_FUNC;
  case static_cast<int64_t>(::llvm::CallingConv::SPIR_KERNEL):
    return ::mlir::LLVM::cconv::CConv::SPIR_KERNEL;
  case static_cast<int64_t>(::llvm::CallingConv::Intel_OCL_BI):
    return ::mlir::LLVM::cconv::CConv::Intel_OCL_BI;
  case static_cast<int64_t>(::llvm::CallingConv::X86_64_SysV):
    return ::mlir::LLVM::cconv::CConv::X86_64_SysV;
  case static_cast<int64_t>(::llvm::CallingConv::Win64):
    return ::mlir::LLVM::cconv::CConv::Win64;
  case static_cast<int64_t>(::llvm::CallingConv::X86_VectorCall):
    return ::mlir::LLVM::cconv::CConv::X86_VectorCall;
  case static_cast<int64_t>(::llvm::CallingConv::HHVM):
    return ::mlir::LLVM::cconv::CConv::HHVM;
  case static_cast<int64_t>(::llvm::CallingConv::HHVM_C):
    return ::mlir::LLVM::cconv::CConv::HHVM_C;
  case static_cast<int64_t>(::llvm::CallingConv::X86_INTR):
    return ::mlir::LLVM::cconv::CConv::X86_INTR;
  case static_cast<int64_t>(::llvm::CallingConv::AVR_INTR):
    return ::mlir::LLVM::cconv::CConv::AVR_INTR;
  case static_cast<int64_t>(::llvm::CallingConv::AVR_BUILTIN):
    return ::mlir::LLVM::cconv::CConv::AVR_BUILTIN;
  case static_cast<int64_t>(::llvm::CallingConv::AMDGPU_VS):
    return ::mlir::LLVM::cconv::CConv::AMDGPU_VS;
  case static_cast<int64_t>(::llvm::CallingConv::AMDGPU_GS):
    return ::mlir::LLVM::cconv::CConv::AMDGPU_GS;
  case static_cast<int64_t>(::llvm::CallingConv::AMDGPU_CS):
    return ::mlir::LLVM::cconv::CConv::AMDGPU_CS;
  case static_cast<int64_t>(::llvm::CallingConv::AMDGPU_KERNEL):
    return ::mlir::LLVM::cconv::CConv::AMDGPU_KERNEL;
  case static_cast<int64_t>(::llvm::CallingConv::X86_RegCall):
    return ::mlir::LLVM::cconv::CConv::X86_RegCall;
  case static_cast<int64_t>(::llvm::CallingConv::AMDGPU_HS):
    return ::mlir::LLVM::cconv::CConv::AMDGPU_HS;
  case static_cast<int64_t>(::llvm::CallingConv::MSP430_BUILTIN):
    return ::mlir::LLVM::cconv::CConv::MSP430_BUILTIN;
  case static_cast<int64_t>(::llvm::CallingConv::AMDGPU_LS):
    return ::mlir::LLVM::cconv::CConv::AMDGPU_LS;
  case static_cast<int64_t>(::llvm::CallingConv::AMDGPU_ES):
    return ::mlir::LLVM::cconv::CConv::AMDGPU_ES;
  case static_cast<int64_t>(::llvm::CallingConv::AArch64_VectorCall):
    return ::mlir::LLVM::cconv::CConv::AArch64_VectorCall;
  case static_cast<int64_t>(::llvm::CallingConv::AArch64_SVE_VectorCall):
    return ::mlir::LLVM::cconv::CConv::AArch64_SVE_VectorCall;
  case static_cast<int64_t>(::llvm::CallingConv::WASM_EmscriptenInvoke):
    return ::mlir::LLVM::cconv::CConv::WASM_EmscriptenInvoke;
  case static_cast<int64_t>(::llvm::CallingConv::AMDGPU_Gfx):
    return ::mlir::LLVM::cconv::CConv::AMDGPU_Gfx;
  case static_cast<int64_t>(::llvm::CallingConv::M68k_INTR):
    return ::mlir::LLVM::cconv::CConv::M68k_INTR;
  }
  llvm_unreachable("unknown ::llvm::CallingConv type");}

