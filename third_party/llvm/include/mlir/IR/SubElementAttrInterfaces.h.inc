/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class SubElementAttrInterface;
namespace detail {
struct SubElementAttrInterfaceInterfaceTraits {
  struct Concept {
    void (*walkImmediateSubElements)(const Concept *impl, ::mlir::Attribute , llvm::function_ref<void(mlir::Attribute)>, llvm::function_ref<void(mlir::Type)>);
    ::mlir::Attribute (*replaceImmediateSubElements)(const Concept *impl, ::mlir::Attribute , ::llvm::ArrayRef<::mlir::Attribute>, ::llvm::ArrayRef<::mlir::Type>);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::SubElementAttrInterface;
    Model() : Concept{walkImmediateSubElements, replaceImmediateSubElements} {}

    static inline void walkImmediateSubElements(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn);
    static inline ::mlir::Attribute replaceImmediateSubElements(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> replAttrs, ::llvm::ArrayRef<::mlir::Type> replTypes);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::SubElementAttrInterface;
    FallbackModel() : Concept{walkImmediateSubElements, replaceImmediateSubElements} {}

    static inline void walkImmediateSubElements(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn);
    static inline ::mlir::Attribute replaceImmediateSubElements(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> replAttrs, ::llvm::ArrayRef<::mlir::Type> replTypes);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
    void walkImmediateSubElements(::mlir::Attribute tablegen_opaque_val, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) const;
    ::mlir::Attribute replaceImmediateSubElements(::mlir::Attribute tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> replAttrs, ::llvm::ArrayRef<::mlir::Type> replTypes) const;
  };
};template <typename ConcreteAttr>
struct SubElementAttrInterfaceTrait;

} // namespace detail
class SubElementAttrInterface : public ::mlir::AttributeInterface<SubElementAttrInterface, detail::SubElementAttrInterfaceInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<SubElementAttrInterface, detail::SubElementAttrInterfaceInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::SubElementAttrInterfaceTrait<ConcreteAttr> {};
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
  ::mlir::Attribute replaceImmediateSubElements(::llvm::ArrayRef<::mlir::Attribute> replAttrs, ::llvm::ArrayRef<::mlir::Type> replTypes) const;

    /// Walk all of the held sub-attributes and sub-types.
    void walkSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                         llvm::function_ref<void(mlir::Type)> walkTypesFn);

    /// Recursively replace all of the nested sub-attributes and sub-types using the
    /// provided map functions. Returns nullptr in the case of failure. See
    /// `AttrTypeReplacer` for information on the support replacement function types.
    template <typename... ReplacementFns>
    ::mlir::Attribute replaceSubElements(ReplacementFns &&... replacementFns) {
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
  template <typename ConcreteAttr>
  struct SubElementAttrInterfaceTrait : public ::mlir::AttributeInterface<SubElementAttrInterface, detail::SubElementAttrInterfaceInterfaceTraits>::Trait<ConcreteAttr> {
    /// Walk all of the immediately nested sub-attributes and sub-types. This
    /// method does not recurse into sub elements.
    void walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) const {
      ::mlir::detail::walkImmediateSubElementsImpl(
          (*static_cast<const ConcreteAttr *>(this)), walkAttrsFn, walkTypesFn);
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
    ::mlir::Attribute replaceImmediateSubElements(::llvm::ArrayRef<::mlir::Attribute> replAttrs, ::llvm::ArrayRef<::mlir::Type> replTypes) const {
      return ::mlir::detail::replaceImmediateSubElementsImpl(
           (*static_cast<const ConcreteAttr *>(this)), replAttrs, replTypes);
    }

    /// Walk all of the held sub-attributes and sub-types.
    void walkSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                         llvm::function_ref<void(mlir::Type)> walkTypesFn) {
      SubElementAttrInterface interface((*static_cast<const ConcreteAttr *>(this)));
      interface.walkSubElements(walkAttrsFn, walkTypesFn);
    }

    /// Recursively replace all of the nested sub-attributes and sub-types using the
    /// provided map functions. Returns nullptr in the case of failure. See
    /// `AttrTypeReplacer` for information on the support replacement function types.
    template <typename... ReplacementFns>
    ::mlir::Attribute replaceSubElements(ReplacementFns &&... replacementFns) {
      AttrTypeReplacer replacer;
      (replacer.addReplacement(std::forward<ReplacementFns>(replacementFns)), ...);
      return replacer.replace((*static_cast<const ConcreteAttr *>(this)));
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
template<typename ConcreteAttr>
void detail::SubElementAttrInterfaceInterfaceTraits::Model<ConcreteAttr>::walkImmediateSubElements(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).walkImmediateSubElements(walkAttrsFn, walkTypesFn);
}
template<typename ConcreteAttr>
::mlir::Attribute detail::SubElementAttrInterfaceInterfaceTraits::Model<ConcreteAttr>::replaceImmediateSubElements(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> replAttrs, ::llvm::ArrayRef<::mlir::Type> replTypes) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).replaceImmediateSubElements(replAttrs, replTypes);
}
template<typename ConcreteAttr>
void detail::SubElementAttrInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::walkImmediateSubElements(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) {
  return static_cast<const ConcreteAttr *>(impl)->walkImmediateSubElements(tablegen_opaque_val, walkAttrsFn, walkTypesFn);
}
template<typename ConcreteAttr>
::mlir::Attribute detail::SubElementAttrInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::replaceImmediateSubElements(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> replAttrs, ::llvm::ArrayRef<::mlir::Type> replTypes) {
  return static_cast<const ConcreteAttr *>(impl)->replaceImmediateSubElements(tablegen_opaque_val, replAttrs, replTypes);
}
template<typename ConcreteModel, typename ConcreteAttr>
void detail::SubElementAttrInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::walkImmediateSubElements(::mlir::Attribute tablegen_opaque_val, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) const {
::mlir::detail::walkImmediateSubElementsImpl(
          (tablegen_opaque_val.cast<ConcreteAttr>()), walkAttrsFn, walkTypesFn);
}
template<typename ConcreteModel, typename ConcreteAttr>
::mlir::Attribute detail::SubElementAttrInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::replaceImmediateSubElements(::mlir::Attribute tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> replAttrs, ::llvm::ArrayRef<::mlir::Type> replTypes) const {
return ::mlir::detail::replaceImmediateSubElementsImpl(
           (tablegen_opaque_val.cast<ConcreteAttr>()), replAttrs, replTypes);
}
} // namespace mlir
