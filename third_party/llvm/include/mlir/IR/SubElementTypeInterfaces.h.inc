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
  ::mlir::Type replaceImmediateSubElements(::llvm::ArrayRef<::mlir::Attribute> replAttrs, ::llvm::ArrayRef<::mlir::Type> replTypes) const;

    /// Walk all of the held sub-attributes and sub-types.
    void walkSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                         llvm::function_ref<void(mlir::Type)> walkTypesFn);

    /// Recursively replace all of the nested sub-attributes and sub-types using the
    /// provided map functions. Returns nullptr in the case of failure.
    ::mlir::Type replaceSubElements(
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
    ::mlir::Type replaceSubElements(
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
    ::mlir::Type replaceSubElements(
      mlir::SubElementReplFn<mlir::Attribute> replaceAttrFn) {
      return replaceSubElements(
        replaceAttrFn, [](mlir::Type type) { return type; });
    }
    ::mlir::Type replaceSubElements(
      mlir::SubElementResultReplFn<mlir::Attribute> replaceAttrFn) {
      return replaceSubElements(
        replaceAttrFn,
        [](mlir::Type type) { return std::make_pair(type, WalkResult::advance()); }
      );
    }
    /// Recursively replace all of the nested sub-types using the provided map
    /// function. Returns nullptr in the case of failure.
    ::mlir::Type replaceSubElements(
      mlir::SubElementReplFn<mlir::Type> replaceTypeFn) {
      return replaceSubElements(
        [](mlir::Attribute attr) { return attr; }, replaceTypeFn);
    }
    ::mlir::Type replaceSubElements(
      mlir::SubElementResultReplFn<mlir::Type> replaceTypeFn) {
      return replaceSubElements(
        [](mlir::Attribute attr) { return std::make_pair(attr, WalkResult::advance()); },
        replaceTypeFn
      );
    }
  };
namespace detail {
  template <typename ConcreteType>
  struct SubElementTypeInterfaceTrait : public ::mlir::TypeInterface<SubElementTypeInterface, detail::SubElementTypeInterfaceInterfaceTraits>::Trait<ConcreteType> {

    /// Walk all of the held sub-attributes and sub-types.
    void walkSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                         llvm::function_ref<void(mlir::Type)> walkTypesFn) {
      SubElementTypeInterface interface((*static_cast<const ConcreteType *>(this)));
      interface.walkSubElements(walkAttrsFn, walkTypesFn);
    }

    /// Recursively replace all of the nested sub-attributes and sub-types using the
    /// provided map functions. Returns nullptr in the case of failure.
    ::mlir::Type replaceSubElements(
      mlir::SubElementReplFn<mlir::Attribute> replaceAttrFn,
      mlir::SubElementReplFn<mlir::Type> replaceTypeFn) {
      SubElementTypeInterface interface((*static_cast<const ConcreteType *>(this)));
      return interface.replaceSubElements(replaceAttrFn, replaceTypeFn);
    }
    ::mlir::Type replaceSubElements(
      mlir::SubElementResultReplFn<mlir::Attribute> replaceAttrFn,
      mlir::SubElementResultReplFn<mlir::Type> replaceTypeFn) {
      SubElementTypeInterface interface((*static_cast<const ConcreteType *>(this)));
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
    ::mlir::Type replaceSubElements(
      mlir::SubElementReplFn<mlir::Attribute> replaceAttrFn) {
      return replaceSubElements(
        replaceAttrFn, [](mlir::Type type) { return type; });
    }
    ::mlir::Type replaceSubElements(
      mlir::SubElementResultReplFn<mlir::Attribute> replaceAttrFn) {
      return replaceSubElements(
        replaceAttrFn,
        [](mlir::Type type) { return std::make_pair(type, WalkResult::advance()); }
      );
    }
    /// Recursively replace all of the nested sub-types using the provided map
    /// function. Returns nullptr in the case of failure.
    ::mlir::Type replaceSubElements(
      mlir::SubElementReplFn<mlir::Type> replaceTypeFn) {
      return replaceSubElements(
        [](mlir::Attribute attr) { return attr; }, replaceTypeFn);
    }
    ::mlir::Type replaceSubElements(
      mlir::SubElementResultReplFn<mlir::Type> replaceTypeFn) {
      return replaceSubElements(
        [](mlir::Attribute attr) { return std::make_pair(attr, WalkResult::advance()); },
        replaceTypeFn
      );
    }
  
  };
}// namespace detail
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
} // namespace mlir
