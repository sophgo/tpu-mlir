/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace amdgpu {
::llvm::StringRef stringifyMFMAPermB(MFMAPermB val) {
  switch (val) {
    case MFMAPermB::none: return "none";
    case MFMAPermB::bcast_first_32: return "bcast_first_32";
    case MFMAPermB::bcast_second_32: return "bcast_second_32";
    case MFMAPermB::rotate_16_right: return "rotate_16_right";
    case MFMAPermB::bcast_first_16: return "bcast_first_16";
    case MFMAPermB::bcast_second_16: return "bcast_second_16";
    case MFMAPermB::bcast_third_16: return "bcast_third_16";
    case MFMAPermB::bcast_fourth_16: return "bcast_fourth_16";
  }
  return "";
}

::llvm::Optional<MFMAPermB> symbolizeMFMAPermB(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<MFMAPermB>>(str)
      .Case("none", MFMAPermB::none)
      .Case("bcast_first_32", MFMAPermB::bcast_first_32)
      .Case("bcast_second_32", MFMAPermB::bcast_second_32)
      .Case("rotate_16_right", MFMAPermB::rotate_16_right)
      .Case("bcast_first_16", MFMAPermB::bcast_first_16)
      .Case("bcast_second_16", MFMAPermB::bcast_second_16)
      .Case("bcast_third_16", MFMAPermB::bcast_third_16)
      .Case("bcast_fourth_16", MFMAPermB::bcast_fourth_16)
      .Default(::llvm::None);
}
::llvm::Optional<MFMAPermB> symbolizeMFMAPermB(uint32_t value) {
  switch (value) {
  case 0: return MFMAPermB::none;
  case 1: return MFMAPermB::bcast_first_32;
  case 2: return MFMAPermB::bcast_second_32;
  case 3: return MFMAPermB::rotate_16_right;
  case 4: return MFMAPermB::bcast_first_16;
  case 5: return MFMAPermB::bcast_second_16;
  case 6: return MFMAPermB::bcast_third_16;
  case 7: return MFMAPermB::bcast_fourth_16;
  default: return ::llvm::None;
  }
}

} // namespace amdgpu
} // namespace mlir

