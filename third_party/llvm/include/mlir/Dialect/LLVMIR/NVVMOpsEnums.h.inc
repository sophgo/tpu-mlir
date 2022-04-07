/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace NVVM {
// MMA binary operations
enum class MMAB1Op : uint32_t {
  none = 0,
  xor_popc = 1,
  and_popc = 2,
};

::llvm::Optional<MMAB1Op> symbolizeMMAB1Op(uint32_t);
::llvm::StringRef stringifyMMAB1Op(MMAB1Op);
::llvm::Optional<MMAB1Op> symbolizeMMAB1Op(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMMAB1Op() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(MMAB1Op enumValue) {
  return stringifyMMAB1Op(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<MMAB1Op> symbolizeEnum<MMAB1Op>(::llvm::StringRef str) {
  return symbolizeMMAB1Op(str);
}
} // namespace NVVM
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::MMAB1Op> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::MMAB1Op getEmptyKey() {
    return static_cast<::mlir::NVVM::MMAB1Op>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::MMAB1Op getTombstoneKey() {
    return static_cast<::mlir::NVVM::MMAB1Op>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::MMAB1Op &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::MMAB1Op &lhs, const ::mlir::NVVM::MMAB1Op &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// NVVM MMA frag type
enum class MMAFrag : uint32_t {
  a = 0,
  b = 1,
  c = 2,
};

::llvm::Optional<MMAFrag> symbolizeMMAFrag(uint32_t);
::llvm::StringRef stringifyMMAFrag(MMAFrag);
::llvm::Optional<MMAFrag> symbolizeMMAFrag(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMMAFrag() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(MMAFrag enumValue) {
  return stringifyMMAFrag(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<MMAFrag> symbolizeEnum<MMAFrag>(::llvm::StringRef str) {
  return symbolizeMMAFrag(str);
}
} // namespace NVVM
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::MMAFrag> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::MMAFrag getEmptyKey() {
    return static_cast<::mlir::NVVM::MMAFrag>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::MMAFrag getTombstoneKey() {
    return static_cast<::mlir::NVVM::MMAFrag>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::MMAFrag &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::MMAFrag &lhs, const ::mlir::NVVM::MMAFrag &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// MMA overflow options
enum class MMAIntOverflow : uint32_t {
  satfinite = 1,
  wrapped = 0,
};

::llvm::Optional<MMAIntOverflow> symbolizeMMAIntOverflow(uint32_t);
::llvm::StringRef stringifyMMAIntOverflow(MMAIntOverflow);
::llvm::Optional<MMAIntOverflow> symbolizeMMAIntOverflow(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMMAIntOverflow() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(MMAIntOverflow enumValue) {
  return stringifyMMAIntOverflow(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<MMAIntOverflow> symbolizeEnum<MMAIntOverflow>(::llvm::StringRef str) {
  return symbolizeMMAIntOverflow(str);
}
} // namespace NVVM
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::MMAIntOverflow> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::MMAIntOverflow getEmptyKey() {
    return static_cast<::mlir::NVVM::MMAIntOverflow>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::MMAIntOverflow getTombstoneKey() {
    return static_cast<::mlir::NVVM::MMAIntOverflow>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::MMAIntOverflow &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::MMAIntOverflow &lhs, const ::mlir::NVVM::MMAIntOverflow &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// NVVM MMA layout
enum class MMALayout : uint32_t {
  row = 0,
  col = 1,
};

::llvm::Optional<MMALayout> symbolizeMMALayout(uint32_t);
::llvm::StringRef stringifyMMALayout(MMALayout);
::llvm::Optional<MMALayout> symbolizeMMALayout(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMMALayout() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(MMALayout enumValue) {
  return stringifyMMALayout(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<MMALayout> symbolizeEnum<MMALayout>(::llvm::StringRef str) {
  return symbolizeMMALayout(str);
}
} // namespace NVVM
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::MMALayout> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::MMALayout getEmptyKey() {
    return static_cast<::mlir::NVVM::MMALayout>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::MMALayout getTombstoneKey() {
    return static_cast<::mlir::NVVM::MMALayout>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::MMALayout &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::MMALayout &lhs, const ::mlir::NVVM::MMALayout &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// NVVM MMA types
enum class MMATypes : uint32_t {
  f16 = 0,
  f32 = 1,
  tf32 = 2,
  bf16 = 9,
  s8 = 4,
  u8 = 3,
  s32 = 5,
  s4 = 8,
  u4 = 7,
  b1 = 6,
  f64 = 10,
};

::llvm::Optional<MMATypes> symbolizeMMATypes(uint32_t);
::llvm::StringRef stringifyMMATypes(MMATypes);
::llvm::Optional<MMATypes> symbolizeMMATypes(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMMATypes() {
  return 10;
}


inline ::llvm::StringRef stringifyEnum(MMATypes enumValue) {
  return stringifyMMATypes(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<MMATypes> symbolizeEnum<MMATypes>(::llvm::StringRef str) {
  return symbolizeMMATypes(str);
}
} // namespace NVVM
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::MMATypes> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::MMATypes getEmptyKey() {
    return static_cast<::mlir::NVVM::MMATypes>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::MMATypes getTombstoneKey() {
    return static_cast<::mlir::NVVM::MMATypes>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::MMATypes &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::MMATypes &lhs, const ::mlir::NVVM::MMATypes &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// NVVM shuffle kind
enum class ShflKind : uint32_t {
  bfly = 0,
  up = 1,
  down = 2,
  idx = 3,
};

::llvm::Optional<ShflKind> symbolizeShflKind(uint32_t);
::llvm::StringRef stringifyShflKind(ShflKind);
::llvm::Optional<ShflKind> symbolizeShflKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForShflKind() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(ShflKind enumValue) {
  return stringifyShflKind(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ShflKind> symbolizeEnum<ShflKind>(::llvm::StringRef str) {
  return symbolizeShflKind(str);
}
} // namespace NVVM
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::ShflKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::ShflKind getEmptyKey() {
    return static_cast<::mlir::NVVM::ShflKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::ShflKind getTombstoneKey() {
    return static_cast<::mlir::NVVM::ShflKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::ShflKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::ShflKind &lhs, const ::mlir::NVVM::ShflKind &rhs) {
    return lhs == rhs;
  }
};
}

