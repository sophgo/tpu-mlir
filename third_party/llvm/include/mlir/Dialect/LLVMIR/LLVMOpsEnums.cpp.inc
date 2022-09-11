/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyAsmDialect(AsmDialect val) {
  switch (val) {
    case AsmDialect::AD_ATT: return "att";
    case AsmDialect::AD_Intel: return "intel";
  }
  return "";
}

::llvm::Optional<AsmDialect> symbolizeAsmDialect(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<AsmDialect>>(str)
      .Case("att", AsmDialect::AD_ATT)
      .Case("intel", AsmDialect::AD_Intel)
      .Default(::llvm::None);
}
::llvm::Optional<AsmDialect> symbolizeAsmDialect(uint64_t value) {
  switch (value) {
  case 0: return AsmDialect::AD_ATT;
  case 1: return AsmDialect::AD_Intel;
  default: return ::llvm::None;
  }
}

bool AsmDialectAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)));
}
AsmDialectAttr AsmDialectAttr::get(::mlir::MLIRContext *context, AsmDialect val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<AsmDialectAttr>();
}
AsmDialect AsmDialectAttr::getValue() const {
  return static_cast<AsmDialect>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyAtomicBinOp(AtomicBinOp val) {
  switch (val) {
    case AtomicBinOp::xchg: return "xchg";
    case AtomicBinOp::add: return "add";
    case AtomicBinOp::sub: return "sub";
    case AtomicBinOp::_and: return "_and";
    case AtomicBinOp::nand: return "nand";
    case AtomicBinOp::_or: return "_or";
    case AtomicBinOp::_xor: return "_xor";
    case AtomicBinOp::max: return "max";
    case AtomicBinOp::min: return "min";
    case AtomicBinOp::umax: return "umax";
    case AtomicBinOp::umin: return "umin";
    case AtomicBinOp::fadd: return "fadd";
    case AtomicBinOp::fsub: return "fsub";
  }
  return "";
}

::llvm::Optional<AtomicBinOp> symbolizeAtomicBinOp(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<AtomicBinOp>>(str)
      .Case("xchg", AtomicBinOp::xchg)
      .Case("add", AtomicBinOp::add)
      .Case("sub", AtomicBinOp::sub)
      .Case("_and", AtomicBinOp::_and)
      .Case("nand", AtomicBinOp::nand)
      .Case("_or", AtomicBinOp::_or)
      .Case("_xor", AtomicBinOp::_xor)
      .Case("max", AtomicBinOp::max)
      .Case("min", AtomicBinOp::min)
      .Case("umax", AtomicBinOp::umax)
      .Case("umin", AtomicBinOp::umin)
      .Case("fadd", AtomicBinOp::fadd)
      .Case("fsub", AtomicBinOp::fsub)
      .Default(::llvm::None);
}
::llvm::Optional<AtomicBinOp> symbolizeAtomicBinOp(uint64_t value) {
  switch (value) {
  case 0: return AtomicBinOp::xchg;
  case 1: return AtomicBinOp::add;
  case 2: return AtomicBinOp::sub;
  case 3: return AtomicBinOp::_and;
  case 4: return AtomicBinOp::nand;
  case 5: return AtomicBinOp::_or;
  case 6: return AtomicBinOp::_xor;
  case 7: return AtomicBinOp::max;
  case 8: return AtomicBinOp::min;
  case 9: return AtomicBinOp::umax;
  case 10: return AtomicBinOp::umin;
  case 11: return AtomicBinOp::fadd;
  case 12: return AtomicBinOp::fsub;
  default: return ::llvm::None;
  }
}

bool AtomicBinOpAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 9)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 10)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 11)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 12)));
}
AtomicBinOpAttr AtomicBinOpAttr::get(::mlir::MLIRContext *context, AtomicBinOp val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<AtomicBinOpAttr>();
}
AtomicBinOp AtomicBinOpAttr::getValue() const {
  return static_cast<AtomicBinOp>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyAtomicOrdering(AtomicOrdering val) {
  switch (val) {
    case AtomicOrdering::not_atomic: return "not_atomic";
    case AtomicOrdering::unordered: return "unordered";
    case AtomicOrdering::monotonic: return "monotonic";
    case AtomicOrdering::acquire: return "acquire";
    case AtomicOrdering::release: return "release";
    case AtomicOrdering::acq_rel: return "acq_rel";
    case AtomicOrdering::seq_cst: return "seq_cst";
  }
  return "";
}

::llvm::Optional<AtomicOrdering> symbolizeAtomicOrdering(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<AtomicOrdering>>(str)
      .Case("not_atomic", AtomicOrdering::not_atomic)
      .Case("unordered", AtomicOrdering::unordered)
      .Case("monotonic", AtomicOrdering::monotonic)
      .Case("acquire", AtomicOrdering::acquire)
      .Case("release", AtomicOrdering::release)
      .Case("acq_rel", AtomicOrdering::acq_rel)
      .Case("seq_cst", AtomicOrdering::seq_cst)
      .Default(::llvm::None);
}
::llvm::Optional<AtomicOrdering> symbolizeAtomicOrdering(uint64_t value) {
  switch (value) {
  case 0: return AtomicOrdering::not_atomic;
  case 1: return AtomicOrdering::unordered;
  case 2: return AtomicOrdering::monotonic;
  case 4: return AtomicOrdering::acquire;
  case 5: return AtomicOrdering::release;
  case 6: return AtomicOrdering::acq_rel;
  case 7: return AtomicOrdering::seq_cst;
  default: return ::llvm::None;
  }
}

bool AtomicOrderingAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)));
}
AtomicOrderingAttr AtomicOrderingAttr::get(::mlir::MLIRContext *context, AtomicOrdering val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<AtomicOrderingAttr>();
}
AtomicOrdering AtomicOrderingAttr::getValue() const {
  return static_cast<AtomicOrdering>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
namespace cconv {
::llvm::StringRef stringifyCConv(CConv val) {
  switch (val) {
    case CConv::C: return "ccc";
    case CConv::Fast: return "fastcc";
    case CConv::Cold: return "coldcc";
    case CConv::GHC: return "cc_10";
    case CConv::HiPE: return "cc_11";
    case CConv::WebKit_JS: return "webkit_jscc";
    case CConv::AnyReg: return "anyregcc";
    case CConv::PreserveMost: return "preserve_mostcc";
    case CConv::PreserveAll: return "preserve_allcc";
    case CConv::Swift: return "swiftcc";
    case CConv::CXX_FAST_TLS: return "cxx_fast_tlscc";
    case CConv::Tail: return "tailcc";
    case CConv::CFGuard_Check: return "cfguard_checkcc";
    case CConv::SwiftTail: return "swifttailcc";
    case CConv::X86_StdCall: return "x86_stdcallcc";
    case CConv::X86_FastCall: return "x86_fastcallcc";
    case CConv::ARM_APCS: return "arm_apcscc";
    case CConv::ARM_AAPCS: return "arm_aapcscc";
    case CConv::ARM_AAPCS_VFP: return "arm_aapcs_vfpcc";
    case CConv::MSP430_INTR: return "msp430_intrcc";
    case CConv::X86_ThisCall: return "x86_thiscallcc";
    case CConv::PTX_Kernel: return "ptx_kernelcc";
    case CConv::PTX_Device: return "ptx_devicecc";
    case CConv::SPIR_FUNC: return "spir_funccc";
    case CConv::SPIR_KERNEL: return "spir_kernelcc";
    case CConv::Intel_OCL_BI: return "intel_ocl_bicc";
    case CConv::X86_64_SysV: return "x86_64_sysvcc";
    case CConv::Win64: return "win64cc";
    case CConv::X86_VectorCall: return "x86_vectorcallcc";
    case CConv::HHVM: return "hhvmcc";
    case CConv::HHVM_C: return "hhvm_ccc";
    case CConv::X86_INTR: return "x86_intrcc";
    case CConv::AVR_INTR: return "avr_intrcc";
    case CConv::AVR_BUILTIN: return "avr_builtincc";
    case CConv::AMDGPU_VS: return "amdgpu_vscc";
    case CConv::AMDGPU_GS: return "amdgpu_gscc";
    case CConv::AMDGPU_CS: return "amdgpu_cscc";
    case CConv::AMDGPU_KERNEL: return "amdgpu_kernelcc";
    case CConv::X86_RegCall: return "x86_regcallcc";
    case CConv::AMDGPU_HS: return "amdgpu_hscc";
    case CConv::MSP430_BUILTIN: return "msp430_builtincc";
    case CConv::AMDGPU_LS: return "amdgpu_lscc";
    case CConv::AMDGPU_ES: return "amdgpu_escc";
    case CConv::AArch64_VectorCall: return "aarch64_vectorcallcc";
    case CConv::AArch64_SVE_VectorCall: return "aarch64_sve_vectorcallcc";
    case CConv::WASM_EmscriptenInvoke: return "wasm_emscripten_invokecc";
    case CConv::AMDGPU_Gfx: return "amdgpu_gfxcc";
    case CConv::M68k_INTR: return "m68k_intrcc";
  }
  return "";
}

::llvm::Optional<CConv> symbolizeCConv(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<CConv>>(str)
      .Case("ccc", CConv::C)
      .Case("fastcc", CConv::Fast)
      .Case("coldcc", CConv::Cold)
      .Case("cc_10", CConv::GHC)
      .Case("cc_11", CConv::HiPE)
      .Case("webkit_jscc", CConv::WebKit_JS)
      .Case("anyregcc", CConv::AnyReg)
      .Case("preserve_mostcc", CConv::PreserveMost)
      .Case("preserve_allcc", CConv::PreserveAll)
      .Case("swiftcc", CConv::Swift)
      .Case("cxx_fast_tlscc", CConv::CXX_FAST_TLS)
      .Case("tailcc", CConv::Tail)
      .Case("cfguard_checkcc", CConv::CFGuard_Check)
      .Case("swifttailcc", CConv::SwiftTail)
      .Case("x86_stdcallcc", CConv::X86_StdCall)
      .Case("x86_fastcallcc", CConv::X86_FastCall)
      .Case("arm_apcscc", CConv::ARM_APCS)
      .Case("arm_aapcscc", CConv::ARM_AAPCS)
      .Case("arm_aapcs_vfpcc", CConv::ARM_AAPCS_VFP)
      .Case("msp430_intrcc", CConv::MSP430_INTR)
      .Case("x86_thiscallcc", CConv::X86_ThisCall)
      .Case("ptx_kernelcc", CConv::PTX_Kernel)
      .Case("ptx_devicecc", CConv::PTX_Device)
      .Case("spir_funccc", CConv::SPIR_FUNC)
      .Case("spir_kernelcc", CConv::SPIR_KERNEL)
      .Case("intel_ocl_bicc", CConv::Intel_OCL_BI)
      .Case("x86_64_sysvcc", CConv::X86_64_SysV)
      .Case("win64cc", CConv::Win64)
      .Case("x86_vectorcallcc", CConv::X86_VectorCall)
      .Case("hhvmcc", CConv::HHVM)
      .Case("hhvm_ccc", CConv::HHVM_C)
      .Case("x86_intrcc", CConv::X86_INTR)
      .Case("avr_intrcc", CConv::AVR_INTR)
      .Case("avr_builtincc", CConv::AVR_BUILTIN)
      .Case("amdgpu_vscc", CConv::AMDGPU_VS)
      .Case("amdgpu_gscc", CConv::AMDGPU_GS)
      .Case("amdgpu_cscc", CConv::AMDGPU_CS)
      .Case("amdgpu_kernelcc", CConv::AMDGPU_KERNEL)
      .Case("x86_regcallcc", CConv::X86_RegCall)
      .Case("amdgpu_hscc", CConv::AMDGPU_HS)
      .Case("msp430_builtincc", CConv::MSP430_BUILTIN)
      .Case("amdgpu_lscc", CConv::AMDGPU_LS)
      .Case("amdgpu_escc", CConv::AMDGPU_ES)
      .Case("aarch64_vectorcallcc", CConv::AArch64_VectorCall)
      .Case("aarch64_sve_vectorcallcc", CConv::AArch64_SVE_VectorCall)
      .Case("wasm_emscripten_invokecc", CConv::WASM_EmscriptenInvoke)
      .Case("amdgpu_gfxcc", CConv::AMDGPU_Gfx)
      .Case("m68k_intrcc", CConv::M68k_INTR)
      .Default(::llvm::None);
}
::llvm::Optional<CConv> symbolizeCConv(uint64_t value) {
  switch (value) {
  case 0: return CConv::C;
  case 8: return CConv::Fast;
  case 9: return CConv::Cold;
  case 10: return CConv::GHC;
  case 11: return CConv::HiPE;
  case 12: return CConv::WebKit_JS;
  case 13: return CConv::AnyReg;
  case 14: return CConv::PreserveMost;
  case 15: return CConv::PreserveAll;
  case 16: return CConv::Swift;
  case 17: return CConv::CXX_FAST_TLS;
  case 18: return CConv::Tail;
  case 19: return CConv::CFGuard_Check;
  case 20: return CConv::SwiftTail;
  case 64: return CConv::X86_StdCall;
  case 65: return CConv::X86_FastCall;
  case 66: return CConv::ARM_APCS;
  case 67: return CConv::ARM_AAPCS;
  case 68: return CConv::ARM_AAPCS_VFP;
  case 69: return CConv::MSP430_INTR;
  case 70: return CConv::X86_ThisCall;
  case 71: return CConv::PTX_Kernel;
  case 72: return CConv::PTX_Device;
  case 75: return CConv::SPIR_FUNC;
  case 76: return CConv::SPIR_KERNEL;
  case 77: return CConv::Intel_OCL_BI;
  case 78: return CConv::X86_64_SysV;
  case 79: return CConv::Win64;
  case 80: return CConv::X86_VectorCall;
  case 81: return CConv::HHVM;
  case 82: return CConv::HHVM_C;
  case 83: return CConv::X86_INTR;
  case 84: return CConv::AVR_INTR;
  case 86: return CConv::AVR_BUILTIN;
  case 87: return CConv::AMDGPU_VS;
  case 88: return CConv::AMDGPU_GS;
  case 90: return CConv::AMDGPU_CS;
  case 91: return CConv::AMDGPU_KERNEL;
  case 92: return CConv::X86_RegCall;
  case 93: return CConv::AMDGPU_HS;
  case 94: return CConv::MSP430_BUILTIN;
  case 95: return CConv::AMDGPU_LS;
  case 96: return CConv::AMDGPU_ES;
  case 97: return CConv::AArch64_VectorCall;
  case 98: return CConv::AArch64_SVE_VectorCall;
  case 99: return CConv::WASM_EmscriptenInvoke;
  case 100: return CConv::AMDGPU_Gfx;
  case 101: return CConv::M68k_INTR;
  default: return ::llvm::None;
  }
}

bool CConvAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 9)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 10)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 11)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 12)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 13)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 14)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 15)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 16)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 17)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 18)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 19)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 20)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 64)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 65)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 66)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 67)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 68)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 69)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 70)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 71)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 72)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 75)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 76)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 77)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 78)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 79)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 80)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 81)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 82)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 83)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 84)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 86)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 87)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 88)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 90)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 91)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 92)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 93)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 94)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 95)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 96)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 97)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 98)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 99)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 100)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 101)));
}
CConvAttr CConvAttr::get(::mlir::MLIRContext *context, CConv val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<CConvAttr>();
}
CConv CConvAttr::getValue() const {
  return static_cast<CConv>(::mlir::IntegerAttr::getInt());
}
} // namespace cconv
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyFCmpPredicate(FCmpPredicate val) {
  switch (val) {
    case FCmpPredicate::_false: return "_false";
    case FCmpPredicate::oeq: return "oeq";
    case FCmpPredicate::ogt: return "ogt";
    case FCmpPredicate::oge: return "oge";
    case FCmpPredicate::olt: return "olt";
    case FCmpPredicate::ole: return "ole";
    case FCmpPredicate::one: return "one";
    case FCmpPredicate::ord: return "ord";
    case FCmpPredicate::ueq: return "ueq";
    case FCmpPredicate::ugt: return "ugt";
    case FCmpPredicate::uge: return "uge";
    case FCmpPredicate::ult: return "ult";
    case FCmpPredicate::ule: return "ule";
    case FCmpPredicate::une: return "une";
    case FCmpPredicate::uno: return "uno";
    case FCmpPredicate::_true: return "_true";
  }
  return "";
}

::llvm::Optional<FCmpPredicate> symbolizeFCmpPredicate(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<FCmpPredicate>>(str)
      .Case("_false", FCmpPredicate::_false)
      .Case("oeq", FCmpPredicate::oeq)
      .Case("ogt", FCmpPredicate::ogt)
      .Case("oge", FCmpPredicate::oge)
      .Case("olt", FCmpPredicate::olt)
      .Case("ole", FCmpPredicate::ole)
      .Case("one", FCmpPredicate::one)
      .Case("ord", FCmpPredicate::ord)
      .Case("ueq", FCmpPredicate::ueq)
      .Case("ugt", FCmpPredicate::ugt)
      .Case("uge", FCmpPredicate::uge)
      .Case("ult", FCmpPredicate::ult)
      .Case("ule", FCmpPredicate::ule)
      .Case("une", FCmpPredicate::une)
      .Case("uno", FCmpPredicate::uno)
      .Case("_true", FCmpPredicate::_true)
      .Default(::llvm::None);
}
::llvm::Optional<FCmpPredicate> symbolizeFCmpPredicate(uint64_t value) {
  switch (value) {
  case 0: return FCmpPredicate::_false;
  case 1: return FCmpPredicate::oeq;
  case 2: return FCmpPredicate::ogt;
  case 3: return FCmpPredicate::oge;
  case 4: return FCmpPredicate::olt;
  case 5: return FCmpPredicate::ole;
  case 6: return FCmpPredicate::one;
  case 7: return FCmpPredicate::ord;
  case 8: return FCmpPredicate::ueq;
  case 9: return FCmpPredicate::ugt;
  case 10: return FCmpPredicate::uge;
  case 11: return FCmpPredicate::ult;
  case 12: return FCmpPredicate::ule;
  case 13: return FCmpPredicate::une;
  case 14: return FCmpPredicate::uno;
  case 15: return FCmpPredicate::_true;
  default: return ::llvm::None;
  }
}

bool FCmpPredicateAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 9)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 10)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 11)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 12)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 13)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 14)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 15)));
}
FCmpPredicateAttr FCmpPredicateAttr::get(::mlir::MLIRContext *context, FCmpPredicate val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<FCmpPredicateAttr>();
}
FCmpPredicate FCmpPredicateAttr::getValue() const {
  return static_cast<FCmpPredicate>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
std::string stringifyFastmathFlags(FastmathFlags symbol) {
  auto val = static_cast<uint32_t>(symbol);
  assert(127u == (127u | val) && "invalid bits set in bit enum");
  ::llvm::SmallVector<::llvm::StringRef, 2> strs;
  // Print bit enum groups before individual bits

  if (127u == (127u & val)) {
    strs.push_back("fast");
    val &= ~static_cast<uint32_t>(127);
  }

  if (1u == (1u & val))
    strs.push_back("nnan");

  if (2u == (2u & val))
    strs.push_back("ninf");

  if (4u == (4u & val))
    strs.push_back("nsz");

  if (8u == (8u & val))
    strs.push_back("arcp");

  if (16u == (16u & val))
    strs.push_back("contract");

  if (32u == (32u & val))
    strs.push_back("afn");

  if (64u == (64u & val))
    strs.push_back("reassoc");
  return ::llvm::join(strs, ", ");
}

::llvm::Optional<FastmathFlags> symbolizeFastmathFlags(::llvm::StringRef str) {
  ::llvm::SmallVector<::llvm::StringRef, 2> symbols;
  str.split(symbols, ",");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<::llvm::Optional<uint32_t>>(symbol.trim())
      .Case("nnan", 1)
      .Case("ninf", 2)
      .Case("nsz", 4)
      .Case("arcp", 8)
      .Case("contract", 16)
      .Case("afn", 32)
      .Case("reassoc", 64)
      .Case("fast", 127)
      .Default(::llvm::None);
    if (bit) { val |= *bit; } else { return ::llvm::None; }
  }
  return static_cast<FastmathFlags>(val);
}

::llvm::Optional<FastmathFlags> symbolizeFastmathFlags(uint32_t value) {
  if (value & ~static_cast<uint32_t>(127u)) return llvm::None;
  return static_cast<FastmathFlags>(value);
}
bool FastmathFlagsAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!(attr.cast<::mlir::IntegerAttr>().getValue().getZExtValue() & (~(1u|2u|4u|8u|16u|32u|64u|127u)))));
}
FastmathFlagsAttr FastmathFlagsAttr::get(::mlir::MLIRContext *context, FastmathFlags val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<FastmathFlagsAttr>();
}
FastmathFlags FastmathFlagsAttr::getValue() const {
  return static_cast<FastmathFlags>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyICmpPredicate(ICmpPredicate val) {
  switch (val) {
    case ICmpPredicate::eq: return "eq";
    case ICmpPredicate::ne: return "ne";
    case ICmpPredicate::slt: return "slt";
    case ICmpPredicate::sle: return "sle";
    case ICmpPredicate::sgt: return "sgt";
    case ICmpPredicate::sge: return "sge";
    case ICmpPredicate::ult: return "ult";
    case ICmpPredicate::ule: return "ule";
    case ICmpPredicate::ugt: return "ugt";
    case ICmpPredicate::uge: return "uge";
  }
  return "";
}

::llvm::Optional<ICmpPredicate> symbolizeICmpPredicate(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ICmpPredicate>>(str)
      .Case("eq", ICmpPredicate::eq)
      .Case("ne", ICmpPredicate::ne)
      .Case("slt", ICmpPredicate::slt)
      .Case("sle", ICmpPredicate::sle)
      .Case("sgt", ICmpPredicate::sgt)
      .Case("sge", ICmpPredicate::sge)
      .Case("ult", ICmpPredicate::ult)
      .Case("ule", ICmpPredicate::ule)
      .Case("ugt", ICmpPredicate::ugt)
      .Case("uge", ICmpPredicate::uge)
      .Default(::llvm::None);
}
::llvm::Optional<ICmpPredicate> symbolizeICmpPredicate(uint64_t value) {
  switch (value) {
  case 0: return ICmpPredicate::eq;
  case 1: return ICmpPredicate::ne;
  case 2: return ICmpPredicate::slt;
  case 3: return ICmpPredicate::sle;
  case 4: return ICmpPredicate::sgt;
  case 5: return ICmpPredicate::sge;
  case 6: return ICmpPredicate::ult;
  case 7: return ICmpPredicate::ule;
  case 8: return ICmpPredicate::ugt;
  case 9: return ICmpPredicate::uge;
  default: return ::llvm::None;
  }
}

bool ICmpPredicateAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 9)));
}
ICmpPredicateAttr ICmpPredicateAttr::get(::mlir::MLIRContext *context, ICmpPredicate val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<ICmpPredicateAttr>();
}
ICmpPredicate ICmpPredicateAttr::getValue() const {
  return static_cast<ICmpPredicate>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
namespace linkage {
::llvm::StringRef stringifyLinkage(Linkage val) {
  switch (val) {
    case Linkage::Private: return "private";
    case Linkage::Internal: return "internal";
    case Linkage::AvailableExternally: return "available_externally";
    case Linkage::Linkonce: return "linkonce";
    case Linkage::Weak: return "weak";
    case Linkage::Common: return "common";
    case Linkage::Appending: return "appending";
    case Linkage::ExternWeak: return "extern_weak";
    case Linkage::LinkonceODR: return "linkonce_odr";
    case Linkage::WeakODR: return "weak_odr";
    case Linkage::External: return "external";
  }
  return "";
}

::llvm::Optional<Linkage> symbolizeLinkage(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<Linkage>>(str)
      .Case("private", Linkage::Private)
      .Case("internal", Linkage::Internal)
      .Case("available_externally", Linkage::AvailableExternally)
      .Case("linkonce", Linkage::Linkonce)
      .Case("weak", Linkage::Weak)
      .Case("common", Linkage::Common)
      .Case("appending", Linkage::Appending)
      .Case("extern_weak", Linkage::ExternWeak)
      .Case("linkonce_odr", Linkage::LinkonceODR)
      .Case("weak_odr", Linkage::WeakODR)
      .Case("external", Linkage::External)
      .Default(::llvm::None);
}
::llvm::Optional<Linkage> symbolizeLinkage(uint64_t value) {
  switch (value) {
  case 0: return Linkage::Private;
  case 1: return Linkage::Internal;
  case 2: return Linkage::AvailableExternally;
  case 3: return Linkage::Linkonce;
  case 4: return Linkage::Weak;
  case 5: return Linkage::Common;
  case 6: return Linkage::Appending;
  case 7: return Linkage::ExternWeak;
  case 8: return Linkage::LinkonceODR;
  case 9: return Linkage::WeakODR;
  case 10: return Linkage::External;
  default: return ::llvm::None;
  }
}

bool LinkageAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 9)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 10)));
}
LinkageAttr LinkageAttr::get(::mlir::MLIRContext *context, Linkage val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<LinkageAttr>();
}
Linkage LinkageAttr::getValue() const {
  return static_cast<Linkage>(::mlir::IntegerAttr::getInt());
}
} // namespace linkage
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyLoopOptionCase(LoopOptionCase val) {
  switch (val) {
    case LoopOptionCase::disable_unroll: return "disable_unroll";
    case LoopOptionCase::disable_licm: return "disable_licm";
    case LoopOptionCase::interleave_count: return "interleave_count";
    case LoopOptionCase::disable_pipeline: return "disable_pipeline";
    case LoopOptionCase::pipeline_initiation_interval: return "pipeline_initiation_interval";
  }
  return "";
}

::llvm::Optional<LoopOptionCase> symbolizeLoopOptionCase(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<LoopOptionCase>>(str)
      .Case("disable_unroll", LoopOptionCase::disable_unroll)
      .Case("disable_licm", LoopOptionCase::disable_licm)
      .Case("interleave_count", LoopOptionCase::interleave_count)
      .Case("disable_pipeline", LoopOptionCase::disable_pipeline)
      .Case("pipeline_initiation_interval", LoopOptionCase::pipeline_initiation_interval)
      .Default(::llvm::None);
}
::llvm::Optional<LoopOptionCase> symbolizeLoopOptionCase(uint32_t value) {
  switch (value) {
  case 1: return LoopOptionCase::disable_unroll;
  case 2: return LoopOptionCase::disable_licm;
  case 3: return LoopOptionCase::interleave_count;
  case 4: return LoopOptionCase::disable_pipeline;
  case 5: return LoopOptionCase::pipeline_initiation_interval;
  default: return ::llvm::None;
  }
}

bool LoopOptionCaseAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 5)));
}
LoopOptionCaseAttr LoopOptionCaseAttr::get(::mlir::MLIRContext *context, LoopOptionCase val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<LoopOptionCaseAttr>();
}
LoopOptionCase LoopOptionCaseAttr::getValue() const {
  return static_cast<LoopOptionCase>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyUnnamedAddr(UnnamedAddr val) {
  switch (val) {
    case UnnamedAddr::None: return "";
    case UnnamedAddr::Local: return "local_unnamed_addr";
    case UnnamedAddr::Global: return "unnamed_addr";
  }
  return "";
}

::llvm::Optional<UnnamedAddr> symbolizeUnnamedAddr(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<UnnamedAddr>>(str)
      .Case("", UnnamedAddr::None)
      .Case("local_unnamed_addr", UnnamedAddr::Local)
      .Case("unnamed_addr", UnnamedAddr::Global)
      .Default(::llvm::None);
}
::llvm::Optional<UnnamedAddr> symbolizeUnnamedAddr(uint64_t value) {
  switch (value) {
  case 0: return UnnamedAddr::None;
  case 1: return UnnamedAddr::Local;
  case 2: return UnnamedAddr::Global;
  default: return ::llvm::None;
  }
}

bool UnnamedAddrAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)));
}
UnnamedAddrAttr UnnamedAddrAttr::get(::mlir::MLIRContext *context, UnnamedAddr val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return baseAttr.cast<UnnamedAddrAttr>();
}
UnnamedAddr UnnamedAddrAttr::getValue() const {
  return static_cast<UnnamedAddr>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

