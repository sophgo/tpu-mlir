/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class SubElementTypeInterface;
namespace detail {
struct SubElementTypeInterfaceInterfaceTraits {
  struct Concept {
    void (*walkImmediateSubElements)(const Concept *impl, ::mlir::Type , llvm::function_ref<void(mlir::Attribute)>, llvm::function_ref<void(mlir::Type)>);
    ::mlir::Type (*replaceImmediateSubElements)(const Concept *impl, ::mlir::Type , ::llvm::ArrayRef<::mlir::Attribute>, ::llvm::ArrayRef<::mlir::Type>);
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::SubElementTypeInterface;
    Model() : Concept{walkImmediateSubElements, replaceImmediateSubElements} {}

    static inline void walkImmediateSubElements(const Concept *impl, ::mlir::Type tablegen_opaque_val, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn);
    static inline ::mlir::Type replaceImmediateSubElements(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> replAttrs, ::llvm::ArrayRef<::mlir::Type> replTypes);
  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::SubElementTypeInterface;
    FallbackModel() : Concept{walkImmediateSubElements, replaceImmediateSubElements} {}

    static inline void walkImmediateSubElements(const Concept *impl, ::mlir::Type tablegen_opaque_val, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn);
    static inline ::mlir::Type replaceImmediateSubElements(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> replAttrs, ::llvm::ArrayRef<::mlir::Type> replTypes);
  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
    void walkImmediateSubElements(::mlir::Type tablegen_opaque_val, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) const;
    ::mlir::Type replaceImmediateSubElements(::mlir::Type tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> replAttrs, ::llvm::ArrayRef<::mlir::Type> replTypes) const;
  };
};template <typename ConcreteType>
struct SubElementTypeInterfaceTrait;

} // namespace detail
class SubElementTypeInterface : public ::mlir::TypeInterface<SubElementTypeInterface, detail::SubElementTypeInterfaceInterfaceTraits> {
public:
  using ::mlir::TypeInterface<SubElementTypeInterface, detail::SubElementTypeInterfaceInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::SubElementTypeInterfaceTrait<ConcreteType> {};
  /// Walk all of the immediately nested sub-attributes and sub-types. This
  /// method does not recurse into sub elements.
  void walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) const;
  /// Replace the immediately nested sub-attributes and sub-types with those provided.
  /// The order of the provided elements is derived from the order of the elements
  /// returned by the callbacks of `walkImmediateSubElements`. The element at index 0
  /// would replace the very first attribute given by `walkImmediateSubElements`.
  /// On success, the new instance with the values replaced is returned. If replacement
  /// fails, nullptr is returned.
  /// 
  /// Note that replacing the sub-elements of mutable types or attributes is
  /// not currently supported by the interface. If an implementing type or
  /// attribute is mutable, it should return `nullptr` if it has no mechanism
  /// for replacing sub elements.
  ::mlir::Type replaceImmediateSubElements(::llvm::ArrayRef<::mlir::Attribute> replAttrs, ::llvm::ArrayRef<::mlir::Type> replTypes) const;

    /// Walk all of the held sub-attributes and sub-types.
    void walkSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                         llvm::function_ref<void(mlir::Type)> walkTypesFn);

    /// Recursively replace all of the nested sub-attributes and sub-types using the
    /// provided map functions. Returns nullptr in the case of failure. See
    /// `AttrTypeReplacer` for information on the support replacement function types.
    template <typename... ReplacementFns>
    ::mlir::Type replaceSubElements(ReplacementFns &&... replacementFns) {
      AttrTypeReplacer replacer;
      (replacer.addReplacement(std::forward<ReplacementFns>(replacementFns)), ...);
      return replacer.replace(*this);
    }
  

    /// Walk all of the held sub-attributes.
    void walkSubAttrs(llvm::function_ref<void(mlir::Attribute)> walkFn) {
      walkSubElements(walkFn, /*walkTypesFn=*/[](mlir::Type) {});
    }
    /// Walk all of the held sub-types.
    void walkSubTypes(llvm::function_ref<void(mlir::Type)> walkFn) {
      walkSubElements(/*walkAttrsFn=*/[](mlir::Attribute) {}, walkFn);
    }
  };
namespace detail {
  template <typename ConcreteType>
  struct SubElementTypeInterfaceTrait : public ::mlir::TypeInterface<SubElementTypeInterface, detail::SubElementTypeInterfaceInterfaceTraits>::Trait<ConcreteType> {
    /// Walk all of the immediately nested sub-attributes and sub-types. This
    /// method does not recurse into sub elements.
    void walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) const {
      ::mlir::detail::walkImmediateSubElementsImpl(
          (*static_cast<const ConcreteType *>(this)), walkAttrsFn, walkTypesFn);
    }
    /// Replace the immediately nested sub-attributes and sub-types with those provided.
    /// The order of the provided elements is derived from the order of the elements
    /// returned by the callbacks of `walkImmediateSubElements`. The element at index 0
    /// would replace the very first attribute given by `walkImmediateSubElements`.
    /// On success, the new instance with the values replaced is returned. If replacement
    /// fails, nullptr is returned.
    /// 
    /// Note that replacing the sub-elements of mutable types or attributes is
    /// not currently supported by the interface. If an implementing type or
    /// attribute is mutable, it should return `nullptr` if it has no mechanism
    /// for replacing sub elements.
    ::mlir::Type replaceImmediateSubElements(::llvm::ArrayRef<::mlir::Attribute> replAttrs, ::llvm::ArrayRef<::mlir::Type> replTypes) const {
      return ::mlir::detail::replaceImmediateSubElementsImpl(
           (*static_cast<const ConcreteType *>(this)), replAttrs, replTypes);
    }

    /// Walk all of the held sub-attributes and sub-types.
    void walkSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                         llvm::function_ref<void(mlir::Type)> walkTypesFn) {
      SubElementTypeInterface interface((*static_cast<const ConcreteType *>(this)));
      interface.walkSubElements(walkAttrsFn, walkTypesFn);
    }

    /// Recursively replace all of the nested sub-attributes and sub-types using the
    /// provided map functions. Returns nullptr in the case of failure. See
    /// `AttrTypeReplacer` for information on the support replacement function types.
    template <typename... ReplacementFns>
    ::mlir::Type replaceSubElements(ReplacementFns &&... replacementFns) {
      AttrTypeReplacer replacer;
      (replacer.addReplacement(std::forward<ReplacementFns>(replacementFns)), ...);
      return replacer.replace((*static_cast<const ConcreteType *>(this)));
    }
  

    /// Walk all of the held sub-attributes.
    void walkSubAttrs(llvm::function_ref<void(mlir::Attribute)> walkFn) {
      walkSubElements(walkFn, /*walkTypesFn=*/[](mlir::Type) {});
    }
    /// Walk all of the held sub-types.
    void walkSubTypes(llvm::function_ref<void(mlir::Type)> walkFn) {
      walkSubElements(/*walkAttrsFn=*/[](mlir::Attribute) {}, walkFn);
    }
  
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteType>
void detail::SubElementTypeInterfaceInterfaceTraits::Model<ConcreteType>::walkImmediateSubElements(const Concept *impl, ::mlir::Type tablegen_opaque_val, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) {
  return (tablegen_opaque_val.cast<ConcreteType>()).walkImmediateSubElements(walkAttrsFn, walkTypesFn);
}
template<typename ConcreteType>
::mlir::Type detail::SubElementTypeInterfaceInterfaceTraits::Model<ConcreteType>::replaceImmediateSubElements(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> replAttrs, ::llvm::ArrayRef<::mlir::Type> replTypes) {
  return (tablegen_opaque_val.cast<ConcreteType>()).replaceImmediateSubElements(replAttrs, replTypes);
}
template<typename ConcreteType>
void detail::SubElementTypeInterfaceInterfaceTraits::FallbackModel<ConcreteType>::walkImmediateSubElements(const Concept *impl, ::mlir::Type tablegen_opaque_val, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) {
  return static_cast<const ConcreteType *>(impl)->walkImmediateSubElements(tablegen_opaque_val, walkAttrsFn, walkTypesFn);
}
template<typename ConcreteType>
::mlir::Type detail::SubElementTypeInterfaceInterfaceTraits::FallbackModel<ConcreteType>::replaceImmediateSubElements(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> replAttrs, ::llvm::ArrayRef<::mlir::Type> replTypes) {
  return static_cast<const ConcreteType *>(impl)->replaceImmediateSubElements(tablegen_opaque_val, replAttrs, replTypes);
}
template<typename ConcreteModel, typename ConcreteType>
void detail::SubElementTypeInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::walkImmediateSubElements(::mlir::Type tablegen_opaque_val, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) const {
::mlir::detail::walkImmediateSubElementsImpl(
          (tablegen_opaque_val.cast<ConcreteType>()), walkAttrsFn, walkTypesFn);
}
template<typename ConcreteModel, typename ConcreteType>
::mlir::Type detail::SubElementTypeInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::replaceImmediateSubElements(::mlir::Type tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> replAttrs, ::llvm::ArrayRef<::mlir::Type> replTypes) const {
return ::mlir::detail::replaceImmediateSubElementsImpl(
           (tablegen_opaque_val.cast<ConcreteType>()), replAttrs, replTypes);
}
} // namespace mlir
