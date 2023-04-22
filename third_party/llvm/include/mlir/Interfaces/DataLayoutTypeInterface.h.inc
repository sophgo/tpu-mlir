/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class DataLayoutTypeInterface;
namespace detail {
struct DataLayoutTypeInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    unsigned (*getTypeSize)(const Concept *impl, ::mlir::Type , const ::mlir::DataLayout &, ::mlir::DataLayoutEntryListRef);
    unsigned (*getTypeSizeInBits)(const Concept *impl, ::mlir::Type , const ::mlir::DataLayout &, ::mlir::DataLayoutEntryListRef);
    unsigned (*getABIAlignment)(const Concept *impl, ::mlir::Type , const ::mlir::DataLayout &, ::mlir::DataLayoutEntryListRef);
    unsigned (*getPreferredAlignment)(const Concept *impl, ::mlir::Type , const ::mlir::DataLayout &, ::mlir::DataLayoutEntryListRef);
    bool (*areCompatible)(const Concept *impl, ::mlir::Type , ::mlir::DataLayoutEntryListRef, ::mlir::DataLayoutEntryListRef);
    ::mlir::LogicalResult (*verifyEntries)(const Concept *impl, ::mlir::Type , ::mlir::DataLayoutEntryListRef, ::mlir::Location);
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::DataLayoutTypeInterface;
    Model() : Concept{getTypeSize, getTypeSizeInBits, getABIAlignment, getPreferredAlignment, areCompatible, verifyEntries} {}

    static inline unsigned getTypeSize(const Concept *impl, ::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getTypeSizeInBits(const Concept *impl, ::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getABIAlignment(const Concept *impl, ::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getPreferredAlignment(const Concept *impl, ::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline bool areCompatible(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::DataLayoutEntryListRef oldLayout, ::mlir::DataLayoutEntryListRef newLayout);
    static inline ::mlir::LogicalResult verifyEntries(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::DataLayoutEntryListRef entries, ::mlir::Location loc);
  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::DataLayoutTypeInterface;
    FallbackModel() : Concept{getTypeSize, getTypeSizeInBits, getABIAlignment, getPreferredAlignment, areCompatible, verifyEntries} {}

    static inline unsigned getTypeSize(const Concept *impl, ::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getTypeSizeInBits(const Concept *impl, ::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getABIAlignment(const Concept *impl, ::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getPreferredAlignment(const Concept *impl, ::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline bool areCompatible(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::DataLayoutEntryListRef oldLayout, ::mlir::DataLayoutEntryListRef newLayout);
    static inline ::mlir::LogicalResult verifyEntries(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::DataLayoutEntryListRef entries, ::mlir::Location loc);
  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
    unsigned getTypeSize(::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout &dataLayout, ::mlir::DataLayoutEntryListRef params) const;
    bool areCompatible(::mlir::Type tablegen_opaque_val, ::mlir::DataLayoutEntryListRef oldLayout, ::mlir::DataLayoutEntryListRef newLayout) const;
    ::mlir::LogicalResult verifyEntries(::mlir::Type tablegen_opaque_val, ::mlir::DataLayoutEntryListRef entries, ::mlir::Location loc) const;
  };
};template <typename ConcreteType>
struct DataLayoutTypeInterfaceTrait;

} // namespace detail
class DataLayoutTypeInterface : public ::mlir::TypeInterface<DataLayoutTypeInterface, detail::DataLayoutTypeInterfaceInterfaceTraits> {
public:
  using ::mlir::TypeInterface<DataLayoutTypeInterface, detail::DataLayoutTypeInterfaceInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::DataLayoutTypeInterfaceTrait<ConcreteType> {};
  /// Returns the size of this type in bytes.
  unsigned getTypeSize(const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) const;
  /// Returns the size of this type in bits.
  unsigned getTypeSizeInBits(const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) const;
  /// Returns the ABI-required alignment for this type, in bytes
  unsigned getABIAlignment(const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) const;
  /// Returns the preferred alignment for this type, in bytes.
  unsigned getPreferredAlignment(const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) const;
  /// Returns true if the two lists of entries are compatible, that is, that `newLayout` spec entries can be nested in an op with `oldLayout` spec entries.
  bool areCompatible(::mlir::DataLayoutEntryListRef oldLayout, ::mlir::DataLayoutEntryListRef newLayout) const;
  /// Verifies that the given list of entries is valid for this type.
  ::mlir::LogicalResult verifyEntries(::mlir::DataLayoutEntryListRef entries, ::mlir::Location loc) const;
};
namespace detail {
  template <typename ConcreteType>
  struct DataLayoutTypeInterfaceTrait : public ::mlir::TypeInterface<DataLayoutTypeInterface, detail::DataLayoutTypeInterfaceInterfaceTraits>::Trait<ConcreteType> {
    /// Returns the size of this type in bytes.
    unsigned getTypeSize(const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) const {
      unsigned bits = (*static_cast<const ConcreteType *>(this)).getTypeSizeInBits(dataLayout, params);
        return ::llvm::divideCeil(bits, 8);
    }
    /// Returns true if the two lists of entries are compatible, that is, that `newLayout` spec entries can be nested in an op with `oldLayout` spec entries.
    bool areCompatible(::mlir::DataLayoutEntryListRef oldLayout, ::mlir::DataLayoutEntryListRef newLayout) const {
      return true;
    }
    /// Verifies that the given list of entries is valid for this type.
    ::mlir::LogicalResult verifyEntries(::mlir::DataLayoutEntryListRef entries, ::mlir::Location loc) const {
      return ::mlir::success();
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteType>
unsigned detail::DataLayoutTypeInterfaceInterfaceTraits::Model<ConcreteType>::getTypeSize(const Concept *impl, ::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return (tablegen_opaque_val.cast<ConcreteType>()).getTypeSize(dataLayout, params);
}
template<typename ConcreteType>
unsigned detail::DataLayoutTypeInterfaceInterfaceTraits::Model<ConcreteType>::getTypeSizeInBits(const Concept *impl, ::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return (tablegen_opaque_val.cast<ConcreteType>()).getTypeSizeInBits(dataLayout, params);
}
template<typename ConcreteType>
unsigned detail::DataLayoutTypeInterfaceInterfaceTraits::Model<ConcreteType>::getABIAlignment(const Concept *impl, ::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return (tablegen_opaque_val.cast<ConcreteType>()).getABIAlignment(dataLayout, params);
}
template<typename ConcreteType>
unsigned detail::DataLayoutTypeInterfaceInterfaceTraits::Model<ConcreteType>::getPreferredAlignment(const Concept *impl, ::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return (tablegen_opaque_val.cast<ConcreteType>()).getPreferredAlignment(dataLayout, params);
}
template<typename ConcreteType>
bool detail::DataLayoutTypeInterfaceInterfaceTraits::Model<ConcreteType>::areCompatible(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::DataLayoutEntryListRef oldLayout, ::mlir::DataLayoutEntryListRef newLayout) {
  return (tablegen_opaque_val.cast<ConcreteType>()).areCompatible(oldLayout, newLayout);
}
template<typename ConcreteType>
::mlir::LogicalResult detail::DataLayoutTypeInterfaceInterfaceTraits::Model<ConcreteType>::verifyEntries(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::DataLayoutEntryListRef entries, ::mlir::Location loc) {
  return (tablegen_opaque_val.cast<ConcreteType>()).verifyEntries(entries, loc);
}
template<typename ConcreteType>
unsigned detail::DataLayoutTypeInterfaceInterfaceTraits::FallbackModel<ConcreteType>::getTypeSize(const Concept *impl, ::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return static_cast<const ConcreteType *>(impl)->getTypeSize(tablegen_opaque_val, dataLayout, params);
}
template<typename ConcreteType>
unsigned detail::DataLayoutTypeInterfaceInterfaceTraits::FallbackModel<ConcreteType>::getTypeSizeInBits(const Concept *impl, ::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return static_cast<const ConcreteType *>(impl)->getTypeSizeInBits(tablegen_opaque_val, dataLayout, params);
}
template<typename ConcreteType>
unsigned detail::DataLayoutTypeInterfaceInterfaceTraits::FallbackModel<ConcreteType>::getABIAlignment(const Concept *impl, ::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return static_cast<const ConcreteType *>(impl)->getABIAlignment(tablegen_opaque_val, dataLayout, params);
}
template<typename ConcreteType>
unsigned detail::DataLayoutTypeInterfaceInterfaceTraits::FallbackModel<ConcreteType>::getPreferredAlignment(const Concept *impl, ::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return static_cast<const ConcreteType *>(impl)->getPreferredAlignment(tablegen_opaque_val, dataLayout, params);
}
template<typename ConcreteType>
bool detail::DataLayoutTypeInterfaceInterfaceTraits::FallbackModel<ConcreteType>::areCompatible(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::DataLayoutEntryListRef oldLayout, ::mlir::DataLayoutEntryListRef newLayout) {
  return static_cast<const ConcreteType *>(impl)->areCompatible(tablegen_opaque_val, oldLayout, newLayout);
}
template<typename ConcreteType>
::mlir::LogicalResult detail::DataLayoutTypeInterfaceInterfaceTraits::FallbackModel<ConcreteType>::verifyEntries(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::DataLayoutEntryListRef entries, ::mlir::Location loc) {
  return static_cast<const ConcreteType *>(impl)->verifyEntries(tablegen_opaque_val, entries, loc);
}
template<typename ConcreteModel, typename ConcreteType>
unsigned detail::DataLayoutTypeInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::getTypeSize(::mlir::Type tablegen_opaque_val, const ::mlir::DataLayout &dataLayout, ::mlir::DataLayoutEntryListRef params) const {
unsigned bits = (tablegen_opaque_val.cast<ConcreteType>()).getTypeSizeInBits(dataLayout, params);
        return ::llvm::divideCeil(bits, 8);
}
template<typename ConcreteModel, typename ConcreteType>
bool detail::DataLayoutTypeInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::areCompatible(::mlir::Type tablegen_opaque_val, ::mlir::DataLayoutEntryListRef oldLayout, ::mlir::DataLayoutEntryListRef newLayout) const {
return true;
}
template<typename ConcreteModel, typename ConcreteType>
::mlir::LogicalResult detail::DataLayoutTypeInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::verifyEntries(::mlir::Type tablegen_opaque_val, ::mlir::DataLayoutEntryListRef entries, ::mlir::Location loc) const {
return ::mlir::success();
}
} // namespace mlir
