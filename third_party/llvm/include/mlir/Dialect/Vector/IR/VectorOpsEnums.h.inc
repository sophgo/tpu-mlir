/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace vector {
// Kind of combining function for contractions and reductions
enum class CombiningKind : uint32_t {
  ADD = 1,
  MUL = 2,
  MINUI = 4,
  MINSI = 8,
  MINF = 16,
  MAXUI = 32,
  MAXSI = 64,
  MAXF = 128,
  AND = 256,
  OR = 512,
  XOR = 1024,
};

::llvm::Optional<CombiningKind> symbolizeCombiningKind(uint32_t);
std::string stringifyCombiningKind(CombiningKind);
::llvm::Optional<CombiningKind> symbolizeCombiningKind(::llvm::StringRef);

inline constexpr CombiningKind operator|(CombiningKind a, CombiningKind b) {
  return static_cast<CombiningKind>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline constexpr CombiningKind operator&(CombiningKind a, CombiningKind b) {
  return static_cast<CombiningKind>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline constexpr CombiningKind operator^(CombiningKind a, CombiningKind b) {
  return static_cast<CombiningKind>(static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b));
}
inline constexpr CombiningKind operator~(CombiningKind bits) {
  // Ensure only bits that can be present in the enum are set
  return static_cast<CombiningKind>(~static_cast<uint32_t>(bits) & static_cast<uint32_t>(2047u));
}
inline constexpr bool bitEnumContainsAll(CombiningKind bits, CombiningKind bit) {
  return (bits & bit) == bit;
}
inline constexpr bool bitEnumContainsAny(CombiningKind bits, CombiningKind bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}
inline constexpr CombiningKind bitEnumClear(CombiningKind bits, CombiningKind bit) {
  return bits & ~bit;
}
inline constexpr CombiningKind bitEnumSet(CombiningKind bits, CombiningKind bit, /*optional*/bool value=true) {
  return value ? (bits | bit) : bitEnumClear(bits, bit);
}
  
inline std::string stringifyEnum(CombiningKind enumValue) {
  return stringifyCombiningKind(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<CombiningKind> symbolizeEnum<CombiningKind>(::llvm::StringRef str) {
  return symbolizeCombiningKind(str);
}
} // namespace vector
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::vector::CombiningKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::vector::CombiningKind getEmptyKey() {
    return static_cast<::mlir::vector::CombiningKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::vector::CombiningKind getTombstoneKey() {
    return static_cast<::mlir::vector::CombiningKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::vector::CombiningKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::vector::CombiningKind &lhs, const ::mlir::vector::CombiningKind &rhs) {
    return lhs == rhs;
  }
};
}

