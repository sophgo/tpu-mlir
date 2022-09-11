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
  ::mlir::Attribute replaceImmediateSubElements(::llvm::ArrayRef<::mlir::Attribute> replAttrs, ::llvm::ArrayRef<::mlir::Type> replTypes) const;

    /// Walk all of the held sub-attributes and sub-types.
    void walkSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                         llvm::function_ref<void(mlir::Type)> walkTypesFn);

    /// Recursively replace all of the nested sub-attributes and sub-types using the
    /// provided map functions. Returns nullptr in the case of failure.
    ::mlir::Attribute replaceSubElements(
      mlir::SubElementReplFn<mlir::Attribute> replaceAttrFn,
      mlir::SubElementReplFn<mlir::Type> replaceTypeFn
    ) {
      return replaceSubElements(
        [&](Attribute attr) { return std::make_pair(replaceAttrFn(attr), WalkResult::advance()); },
        [&](Type type) { return std::make_pair(replaceTypeFn(type), WalkResult::advance()); }
      );
    }
    /// Recursively replace all of the nested sub-attributes and sub-types using the
    /// provided map functions. This variant allows for the map function to return an
    /// additional walk result. Returns nullptr in the case of failure.
    ::mlir::Attribute replaceSubElements(
      llvm::function_ref<
        std::pair<mlir::Attribute, mlir::WalkResult>(mlir::Attribute)> replaceAttrFn,
      llvm::function_ref<
        std::pair<mlir::Type, mlir::WalkResult>(mlir::Type)> replaceTypeFn
    );
  

    /// Walk all of the held sub-attributes.
    void walkSubAttrs(llvm::function_ref<void(mlir::Attribute)> walkFn) {
      walkSubElements(walkFn, /*walkTypesFn=*/[](mlir::Type) {});
    }
    /// Walk all of the held sub-types.
    void walkSubTypes(llvm::function_ref<void(mlir::Type)> walkFn) {
      walkSubElements(/*walkAttrsFn=*/[](mlir::Attribute) {}, walkFn);
    }
    
    /// Recursively replace all of the nested sub-attributes using the provided
    /// map function. Returns nullptr in the case of failure.
    ::mlir::Attribute replaceSubElements(
      mlir::SubElementReplFn<mlir::Attribute> replaceAttrFn) {
      return replaceSubElements(
        replaceAttrFn, [](mlir::Type type) { return type; });
    }
    ::mlir::Attribute replaceSubElements(
      mlir::SubElementResultReplFn<mlir::Attribute> replaceAttrFn) {
      return replaceSubElements(
        replaceAttrFn,
        [](mlir::Type type) { return std::make_pair(type, WalkResult::advance()); }
      );
    }
    /// Recursively replace all of the nested sub-types using the provided map
    /// function. Returns nullptr in the case of failure.
    ::mlir::Attribute replaceSubElements(
      mlir::SubElementReplFn<mlir::Type> replaceTypeFn) {
      return replaceSubElements(
        [](mlir::Attribute attr) { return attr; }, replaceTypeFn);
    }
    ::mlir::Attribute replaceSubElements(
      mlir::SubElementResultReplFn<mlir::Type> replaceTypeFn) {
      return replaceSubElements(
        [](mlir::Attribute attr) { return std::make_pair(attr, WalkResult::advance()); },
        replaceTypeFn
      );
    }
  };
namespace detail {
  template <typename ConcreteAttr>
  struct SubElementAttrInterfaceTrait : public ::mlir::AttributeInterface<SubElementAttrInterface, detail::SubElementAttrInterfaceInterfaceTraits>::Trait<ConcreteAttr> {

    /// Walk all of the held sub-attributes and sub-types.
    void walkSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                         llvm::function_ref<void(mlir::Type)> walkTypesFn) {
      SubElementAttrInterface interface((*static_cast<const ConcreteAttr *>(this)));
      interface.walkSubElements(walkAttrsFn, walkTypesFn);
    }

    /// Recursively replace all of the nested sub-attributes and sub-types using the
    /// provided map functions. Returns nullptr in the case of failure.
    ::mlir::Attribute replaceSubElements(
      mlir::SubElementReplFn<mlir::Attribute> replaceAttrFn,
      mlir::SubElementReplFn<mlir::Type> replaceTypeFn) {
      SubElementAttrInterface interface((*static_cast<const ConcreteAttr *>(this)));
      return interface.replaceSubElements(replaceAttrFn, replaceTypeFn);
    }
    ::mlir::Attribute replaceSubElements(
      mlir::SubElementResultReplFn<mlir::Attribute> replaceAttrFn,
      mlir::SubElementResultReplFn<mlir::Type> replaceTypeFn) {
      SubElementAttrInterface interface((*static_cast<const ConcreteAttr *>(this)));
      return interface.replaceSubElements(replaceAttrFn, replaceTypeFn);
    }
  

    /// Walk all of the held sub-attributes.
    void walkSubAttrs(llvm::function_ref<void(mlir::Attribute)> walkFn) {
      walkSubElements(walkFn, /*walkTypesFn=*/[](mlir::Type) {});
    }
    /// Walk all of the held sub-types.
    void walkSubTypes(llvm::function_ref<void(mlir::Type)> walkFn) {
      walkSubElements(/*walkAttrsFn=*/[](mlir::Attribute) {}, walkFn);
    }
    
    /// Recursively replace all of the nested sub-attributes using the provided
    /// map function. Returns nullptr in the case of failure.
    ::mlir::Attribute replaceSubElements(
      mlir::SubElementReplFn<mlir::Attribute> replaceAttrFn) {
      return replaceSubElements(
        replaceAttrFn, [](mlir::Type type) { return type; });
    }
    ::mlir::Attribute replaceSubElements(
      mlir::SubElementResultReplFn<mlir::Attribute> replaceAttrFn) {
      return replaceSubElements(
        replaceAttrFn,
        [](mlir::Type type) { return std::make_pair(type, WalkResult::advance()); }
      );
    }
    /// Recursively replace all of the nested sub-types using the provided map
    /// function. Returns nullptr in the case of failure.
    ::mlir::Attribute replaceSubElements(
      mlir::SubElementReplFn<mlir::Type> replaceTypeFn) {
      return replaceSubElements(
        [](mlir::Attribute attr) { return attr; }, replaceTypeFn);
    }
    ::mlir::Attribute replaceSubElements(
      mlir::SubElementResultReplFn<mlir::Type> replaceTypeFn) {
      return replaceSubElements(
        [](mlir::Attribute attr) { return std::make_pair(attr, WalkResult::advance()); },
        replaceTypeFn
      );
    }
  
  };
}// namespace detail
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
} // namespace mlir
