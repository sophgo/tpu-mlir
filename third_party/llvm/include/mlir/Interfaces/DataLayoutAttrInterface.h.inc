/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class DataLayoutEntryInterface;
namespace detail {
struct DataLayoutEntryInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::DataLayoutEntryKey (*getKey)(const Concept *impl, ::mlir::Attribute );
    ::mlir::Attribute (*getValue)(const Concept *impl, ::mlir::Attribute );
    ::mlir::LogicalResult (*verifyEntry)(const Concept *impl, ::mlir::Attribute , ::mlir::Location);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::DataLayoutEntryInterface;
    Model() : Concept{getKey, getValue, verifyEntry} {}

    static inline ::mlir::DataLayoutEntryKey getKey(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline ::mlir::Attribute getValue(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline ::mlir::LogicalResult verifyEntry(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::DataLayoutEntryInterface;
    FallbackModel() : Concept{getKey, getValue, verifyEntry} {}

    static inline ::mlir::DataLayoutEntryKey getKey(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline ::mlir::Attribute getValue(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline ::mlir::LogicalResult verifyEntry(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    ::mlir::LogicalResult verifyEntry(::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) const;
  };
};template <typename ConcreteAttr>
struct DataLayoutEntryInterfaceTrait;

} // namespace detail
class DataLayoutEntryInterface : public ::mlir::AttributeInterface<DataLayoutEntryInterface, detail::DataLayoutEntryInterfaceInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<DataLayoutEntryInterface, detail::DataLayoutEntryInterfaceInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::DataLayoutEntryInterfaceTrait<ConcreteAttr> {};
  ::mlir::DataLayoutEntryKey getKey() const;
  ::mlir::Attribute getValue() const;
  ::mlir::LogicalResult verifyEntry(::mlir::Location loc) const;

    /// Returns `true` if the key of this entry is a type.
    bool isTypeEntry() {
      return getKey().is<::mlir::Type>();
    }
  
};
namespace detail {
  template <typename ConcreteAttr>
  struct DataLayoutEntryInterfaceTrait : public ::mlir::AttributeInterface<DataLayoutEntryInterface, detail::DataLayoutEntryInterfaceInterfaceTraits>::Trait<ConcreteAttr> {
    ::mlir::LogicalResult verifyEntry(::mlir::Location loc) const {
      return ::mlir::success();
    }
  };
}// namespace detail
template<typename ConcreteAttr>
::mlir::DataLayoutEntryKey detail::DataLayoutEntryInterfaceInterfaceTraits::Model<ConcreteAttr>::getKey(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).getKey();
}
template<typename ConcreteAttr>
::mlir::Attribute detail::DataLayoutEntryInterfaceInterfaceTraits::Model<ConcreteAttr>::getValue(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).getValue();
}
template<typename ConcreteAttr>
::mlir::LogicalResult detail::DataLayoutEntryInterfaceInterfaceTraits::Model<ConcreteAttr>::verifyEntry(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).verifyEntry(loc);
}
template<typename ConcreteAttr>
::mlir::DataLayoutEntryKey detail::DataLayoutEntryInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getKey(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getKey(tablegen_opaque_val);
}
template<typename ConcreteAttr>
::mlir::Attribute detail::DataLayoutEntryInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getValue(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getValue(tablegen_opaque_val);
}
template<typename ConcreteAttr>
::mlir::LogicalResult detail::DataLayoutEntryInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::verifyEntry(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) {
  return static_cast<const ConcreteAttr *>(impl)->verifyEntry(tablegen_opaque_val, loc);
}
template<typename ConcreteModel, typename ConcreteAttr>
::mlir::LogicalResult detail::DataLayoutEntryInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::verifyEntry(::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) const {
return ::mlir::success();
}
} // namespace mlir
namespace mlir {
class DataLayoutSpecInterface;
namespace detail {
struct DataLayoutSpecInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::DataLayoutSpecInterface (*combineWith)(const Concept *impl, ::mlir::Attribute , ::llvm::ArrayRef<::mlir::DataLayoutSpecInterface>);
    ::mlir::DataLayoutEntryListRef (*getEntries)(const Concept *impl, ::mlir::Attribute );
    ::mlir::DataLayoutEntryList (*getSpecForType)(const Concept *impl, ::mlir::Attribute , ::mlir::TypeID);
    ::mlir::DataLayoutEntryInterface (*getSpecForIdentifier)(const Concept *impl, ::mlir::Attribute , ::mlir::StringAttr);
    ::mlir::LogicalResult (*verifySpec)(const Concept *impl, ::mlir::Attribute , ::mlir::Location);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::DataLayoutSpecInterface;
    Model() : Concept{combineWith, getEntries, getSpecForType, getSpecForIdentifier, verifySpec} {}

    static inline ::mlir::DataLayoutSpecInterface combineWith(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::llvm::ArrayRef<::mlir::DataLayoutSpecInterface> specs);
    static inline ::mlir::DataLayoutEntryListRef getEntries(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline ::mlir::DataLayoutEntryList getSpecForType(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::TypeID type);
    static inline ::mlir::DataLayoutEntryInterface getSpecForIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::StringAttr identifier);
    static inline ::mlir::LogicalResult verifySpec(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::DataLayoutSpecInterface;
    FallbackModel() : Concept{combineWith, getEntries, getSpecForType, getSpecForIdentifier, verifySpec} {}

    static inline ::mlir::DataLayoutSpecInterface combineWith(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::llvm::ArrayRef<::mlir::DataLayoutSpecInterface> specs);
    static inline ::mlir::DataLayoutEntryListRef getEntries(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline ::mlir::DataLayoutEntryList getSpecForType(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::TypeID type);
    static inline ::mlir::DataLayoutEntryInterface getSpecForIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::StringAttr identifier);
    static inline ::mlir::LogicalResult verifySpec(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    ::mlir::DataLayoutEntryList getSpecForType(::mlir::Attribute tablegen_opaque_val, ::mlir::TypeID type) const;
    ::mlir::DataLayoutEntryInterface getSpecForIdentifier(::mlir::Attribute tablegen_opaque_val, ::mlir::StringAttr identifier) const;
    ::mlir::LogicalResult verifySpec(::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) const;
  };
};template <typename ConcreteAttr>
struct DataLayoutSpecInterfaceTrait;

} // namespace detail
class DataLayoutSpecInterface : public ::mlir::AttributeInterface<DataLayoutSpecInterface, detail::DataLayoutSpecInterfaceInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<DataLayoutSpecInterface, detail::DataLayoutSpecInterfaceInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::DataLayoutSpecInterfaceTrait<ConcreteAttr> {};
  ::mlir::DataLayoutSpecInterface combineWith(::llvm::ArrayRef<::mlir::DataLayoutSpecInterface> specs) const;
  ::mlir::DataLayoutEntryListRef getEntries() const;
  ::mlir::DataLayoutEntryList getSpecForType(::mlir::TypeID type) const;
  ::mlir::DataLayoutEntryInterface getSpecForIdentifier(::mlir::StringAttr identifier) const;
  ::mlir::LogicalResult verifySpec(::mlir::Location loc) const;

    /// Returns a copy of the entries related to the type specified as template
    /// parameter.
    template <typename Ty>
    DataLayoutEntryList getSpecForType() {
      return getSpecForType(TypeID::get<Ty>());
    }

    /// Populates the given maps with lists of entries grouped by the type or
    /// identifier they are associated with. Users are not expected to call this
    /// method directly.
    void bucketEntriesByType(
        ::llvm::DenseMap<::mlir::TypeID, ::mlir::DataLayoutEntryList> &types,
        ::llvm::DenseMap<::mlir::StringAttr,
                         ::mlir::DataLayoutEntryInterface> &ids);
  
};
namespace detail {
  template <typename ConcreteAttr>
  struct DataLayoutSpecInterfaceTrait : public ::mlir::AttributeInterface<DataLayoutSpecInterface, detail::DataLayoutSpecInterfaceInterfaceTraits>::Trait<ConcreteAttr> {
    ::mlir::DataLayoutEntryList getSpecForType(::mlir::TypeID type) const {
      return ::mlir::detail::filterEntriesForType((*static_cast<const ConcreteAttr *>(this)).getEntries(), type);
    }
    ::mlir::DataLayoutEntryInterface getSpecForIdentifier(::mlir::StringAttr identifier) const {
      return ::mlir::detail::filterEntryForIdentifier((*static_cast<const ConcreteAttr *>(this)).getEntries(),
                                                        identifier);
    }
    ::mlir::LogicalResult verifySpec(::mlir::Location loc) const {
      return ::mlir::detail::verifyDataLayoutSpec((*static_cast<const ConcreteAttr *>(this)), loc);
    }
  };
}// namespace detail
template<typename ConcreteAttr>
::mlir::DataLayoutSpecInterface detail::DataLayoutSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::combineWith(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::llvm::ArrayRef<::mlir::DataLayoutSpecInterface> specs) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).combineWith(specs);
}
template<typename ConcreteAttr>
::mlir::DataLayoutEntryListRef detail::DataLayoutSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::getEntries(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).getEntries();
}
template<typename ConcreteAttr>
::mlir::DataLayoutEntryList detail::DataLayoutSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::getSpecForType(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::TypeID type) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).getSpecForType(type);
}
template<typename ConcreteAttr>
::mlir::DataLayoutEntryInterface detail::DataLayoutSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::getSpecForIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::StringAttr identifier) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).getSpecForIdentifier(identifier);
}
template<typename ConcreteAttr>
::mlir::LogicalResult detail::DataLayoutSpecInterfaceInterfaceTraits::Model<ConcreteAttr>::verifySpec(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).verifySpec(loc);
}
template<typename ConcreteAttr>
::mlir::DataLayoutSpecInterface detail::DataLayoutSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::combineWith(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::llvm::ArrayRef<::mlir::DataLayoutSpecInterface> specs) {
  return static_cast<const ConcreteAttr *>(impl)->combineWith(tablegen_opaque_val, specs);
}
template<typename ConcreteAttr>
::mlir::DataLayoutEntryListRef detail::DataLayoutSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getEntries(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getEntries(tablegen_opaque_val);
}
template<typename ConcreteAttr>
::mlir::DataLayoutEntryList detail::DataLayoutSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getSpecForType(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::TypeID type) {
  return static_cast<const ConcreteAttr *>(impl)->getSpecForType(tablegen_opaque_val, type);
}
template<typename ConcreteAttr>
::mlir::DataLayoutEntryInterface detail::DataLayoutSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getSpecForIdentifier(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::StringAttr identifier) {
  return static_cast<const ConcreteAttr *>(impl)->getSpecForIdentifier(tablegen_opaque_val, identifier);
}
template<typename ConcreteAttr>
::mlir::LogicalResult detail::DataLayoutSpecInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::verifySpec(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) {
  return static_cast<const ConcreteAttr *>(impl)->verifySpec(tablegen_opaque_val, loc);
}
template<typename ConcreteModel, typename ConcreteAttr>
::mlir::DataLayoutEntryList detail::DataLayoutSpecInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getSpecForType(::mlir::Attribute tablegen_opaque_val, ::mlir::TypeID type) const {
return ::mlir::detail::filterEntriesForType((tablegen_opaque_val.cast<ConcreteAttr>()).getEntries(), type);
}
template<typename ConcreteModel, typename ConcreteAttr>
::mlir::DataLayoutEntryInterface detail::DataLayoutSpecInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getSpecForIdentifier(::mlir::Attribute tablegen_opaque_val, ::mlir::StringAttr identifier) const {
return ::mlir::detail::filterEntryForIdentifier((tablegen_opaque_val.cast<ConcreteAttr>()).getEntries(),
                                                        identifier);
}
template<typename ConcreteModel, typename ConcreteAttr>
::mlir::LogicalResult detail::DataLayoutSpecInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::verifySpec(::mlir::Attribute tablegen_opaque_val, ::mlir::Location loc) const {
return ::mlir::detail::verifyDataLayoutSpec((tablegen_opaque_val.cast<ConcreteAttr>()), loc);
}
} // namespace mlir
