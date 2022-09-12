/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace amdgpu {
// The possible permutations of the lanes storing B available in an MFMA
enum class MFMAPermB : uint32_t {
  none = 0,
  bcast_first_32 = 1,
  bcast_second_32 = 2,
  rotate_16_right = 3,
  bcast_first_16 = 4,
  bcast_second_16 = 5,
  bcast_third_16 = 6,
  bcast_fourth_16 = 7,
};

::llvm::Optional<MFMAPermB> symbolizeMFMAPermB(uint32_t);
::llvm::StringRef stringifyMFMAPermB(MFMAPermB);
::llvm::Optional<MFMAPermB> symbolizeMFMAPermB(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMFMAPermB() {
  return 7;
}


inline ::llvm::StringRef stringifyEnum(MFMAPermB enumValue) {
  return stringifyMFMAPermB(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<MFMAPermB> symbolizeEnum<MFMAPermB>(::llvm::StringRef str) {
  return symbolizeMFMAPermB(str);
}
} // namespace amdgpu
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::amdgpu::MFMAPermB> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::amdgpu::MFMAPermB getEmptyKey() {
    return static_cast<::mlir::amdgpu::MFMAPermB>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::amdgpu::MFMAPermB getTombstoneKey() {
    return static_cast<::mlir::amdgpu::MFMAPermB>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::amdgpu::MFMAPermB &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::amdgpu::MFMAPermB &lhs, const ::mlir::amdgpu::MFMAPermB &rhs) {
    return lhs == rhs;
  }
};
}

