/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace arith {
// allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
enum class CmpFPredicate : uint64_t {
  AlwaysFalse = 0,
  OEQ = 1,
  OGT = 2,
  OGE = 3,
  OLT = 4,
  OLE = 5,
  ONE = 6,
  ORD = 7,
  UEQ = 8,
  UGT = 9,
  UGE = 10,
  ULT = 11,
  ULE = 12,
  UNE = 13,
  UNO = 14,
  AlwaysTrue = 15,
};

::llvm::Optional<CmpFPredicate> symbolizeCmpFPredicate(uint64_t);
::llvm::StringRef stringifyCmpFPredicate(CmpFPredicate);
::llvm::Optional<CmpFPredicate> symbolizeCmpFPredicate(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCmpFPredicate() {
  return 15;
}


inline ::llvm::StringRef stringifyEnum(CmpFPredicate enumValue) {
  return stringifyCmpFPredicate(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<CmpFPredicate> symbolizeEnum<CmpFPredicate>(::llvm::StringRef str) {
  return symbolizeCmpFPredicate(str);
}

class CmpFPredicateAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = CmpFPredicate;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static CmpFPredicateAttr get(::mlir::MLIRContext *context, CmpFPredicate val);
  CmpFPredicate getValue() const;
};
} // namespace arith
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::arith::CmpFPredicate> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::arith::CmpFPredicate getEmptyKey() {
    return static_cast<::mlir::arith::CmpFPredicate>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::arith::CmpFPredicate getTombstoneKey() {
    return static_cast<::mlir::arith::CmpFPredicate>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::arith::CmpFPredicate &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::arith::CmpFPredicate &lhs, const ::mlir::arith::CmpFPredicate &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace arith {
// allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
enum class CmpIPredicate : uint64_t {
  eq = 0,
  ne = 1,
  slt = 2,
  sle = 3,
  sgt = 4,
  sge = 5,
  ult = 6,
  ule = 7,
  ugt = 8,
  uge = 9,
};

::llvm::Optional<CmpIPredicate> symbolizeCmpIPredicate(uint64_t);
::llvm::StringRef stringifyCmpIPredicate(CmpIPredicate);
::llvm::Optional<CmpIPredicate> symbolizeCmpIPredicate(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCmpIPredicate() {
  return 9;
}


inline ::llvm::StringRef stringifyEnum(CmpIPredicate enumValue) {
  return stringifyCmpIPredicate(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<CmpIPredicate> symbolizeEnum<CmpIPredicate>(::llvm::StringRef str) {
  return symbolizeCmpIPredicate(str);
}

class CmpIPredicateAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = CmpIPredicate;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static CmpIPredicateAttr get(::mlir::MLIRContext *context, CmpIPredicate val);
  CmpIPredicate getValue() const;
};
} // namespace arith
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::arith::CmpIPredicate> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::arith::CmpIPredicate getEmptyKey() {
    return static_cast<::mlir::arith::CmpIPredicate>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::arith::CmpIPredicate getTombstoneKey() {
    return static_cast<::mlir::arith::CmpIPredicate>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::arith::CmpIPredicate &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::arith::CmpIPredicate &lhs, const ::mlir::arith::CmpIPredicate &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace arith {
// allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
enum class AtomicRMWKind : uint64_t {
  addf = 0,
  addi = 1,
  assign = 2,
  maxf = 3,
  maxs = 4,
  maxu = 5,
  minf = 6,
  mins = 7,
  minu = 8,
  mulf = 9,
  muli = 10,
  ori = 11,
  andi = 12,
};

::llvm::Optional<AtomicRMWKind> symbolizeAtomicRMWKind(uint64_t);
::llvm::StringRef stringifyAtomicRMWKind(AtomicRMWKind);
::llvm::Optional<AtomicRMWKind> symbolizeAtomicRMWKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAtomicRMWKind() {
  return 12;
}


inline ::llvm::StringRef stringifyEnum(AtomicRMWKind enumValue) {
  return stringifyAtomicRMWKind(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<AtomicRMWKind> symbolizeEnum<AtomicRMWKind>(::llvm::StringRef str) {
  return symbolizeAtomicRMWKind(str);
}

class AtomicRMWKindAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = AtomicRMWKind;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static AtomicRMWKindAttr get(::mlir::MLIRContext *context, AtomicRMWKind val);
  AtomicRMWKind getValue() const;
};
} // namespace arith
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::arith::AtomicRMWKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::arith::AtomicRMWKind getEmptyKey() {
    return static_cast<::mlir::arith::AtomicRMWKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::arith::AtomicRMWKind getTombstoneKey() {
    return static_cast<::mlir::arith::AtomicRMWKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::arith::AtomicRMWKind &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::arith::AtomicRMWKind &lhs, const ::mlir::arith::AtomicRMWKind &rhs) {
    return lhs == rhs;
  }
};
}

