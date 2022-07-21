/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class SymbolOpInterface;
namespace detail {
struct SymbolOpInterfaceInterfaceTraits {
  struct Concept {
    StringAttr (*getNameAttr)(const Concept *impl, ::mlir::Operation *);
    void (*setName)(const Concept *impl, ::mlir::Operation *, StringAttr);
    mlir::SymbolTable::Visibility (*getVisibility)(const Concept *impl, ::mlir::Operation *);
    bool (*isNested)(const Concept *impl, ::mlir::Operation *);
    bool (*isPrivate)(const Concept *impl, ::mlir::Operation *);
    bool (*isPublic)(const Concept *impl, ::mlir::Operation *);
    void (*setVisibility)(const Concept *impl, ::mlir::Operation *, mlir::SymbolTable::Visibility);
    void (*setNested)(const Concept *impl, ::mlir::Operation *);
    void (*setPrivate)(const Concept *impl, ::mlir::Operation *);
    void (*setPublic)(const Concept *impl, ::mlir::Operation *);
    Optional<::mlir::SymbolTable::UseRange> (*getSymbolUses)(const Concept *impl, ::mlir::Operation *, Operation *);
    bool (*symbolKnownUseEmpty)(const Concept *impl, ::mlir::Operation *, Operation *);
    LogicalResult (*replaceAllSymbolUses)(const Concept *impl, ::mlir::Operation *, StringAttr, Operation *);
    bool (*isOptionalSymbol)(const Concept *impl, ::mlir::Operation *);
    bool (*canDiscardOnUseEmpty)(const Concept *impl, ::mlir::Operation *);
    bool (*isDeclaration)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::SymbolOpInterface;
    Model() : Concept{getNameAttr, setName, getVisibility, isNested, isPrivate, isPublic, setVisibility, setNested, setPrivate, setPublic, getSymbolUses, symbolKnownUseEmpty, replaceAllSymbolUses, isOptionalSymbol, canDiscardOnUseEmpty, isDeclaration} {}

    static inline StringAttr getNameAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, StringAttr name);
    static inline mlir::SymbolTable::Visibility getVisibility(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isNested(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isPrivate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isPublic(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setVisibility(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::SymbolTable::Visibility vis);
    static inline void setNested(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setPrivate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setPublic(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Optional<::mlir::SymbolTable::UseRange> getSymbolUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Operation * from);
    static inline bool symbolKnownUseEmpty(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Operation * from);
    static inline LogicalResult replaceAllSymbolUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, StringAttr newSymbol, Operation * from);
    static inline bool isOptionalSymbol(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool canDiscardOnUseEmpty(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isDeclaration(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::SymbolOpInterface;
    FallbackModel() : Concept{getNameAttr, setName, getVisibility, isNested, isPrivate, isPublic, setVisibility, setNested, setPrivate, setPublic, getSymbolUses, symbolKnownUseEmpty, replaceAllSymbolUses, isOptionalSymbol, canDiscardOnUseEmpty, isDeclaration} {}

    static inline StringAttr getNameAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, StringAttr name);
    static inline mlir::SymbolTable::Visibility getVisibility(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isNested(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isPrivate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isPublic(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setVisibility(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::SymbolTable::Visibility vis);
    static inline void setNested(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setPrivate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setPublic(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Optional<::mlir::SymbolTable::UseRange> getSymbolUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Operation * from);
    static inline bool symbolKnownUseEmpty(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Operation * from);
    static inline LogicalResult replaceAllSymbolUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, StringAttr newSymbol, Operation * from);
    static inline bool isOptionalSymbol(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool canDiscardOnUseEmpty(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isDeclaration(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    StringAttr getNameAttr(::mlir::Operation *tablegen_opaque_val) const;
    void setName(::mlir::Operation *tablegen_opaque_val, StringAttr name) const;
    mlir::SymbolTable::Visibility getVisibility(::mlir::Operation *tablegen_opaque_val) const;
    bool isNested(::mlir::Operation *tablegen_opaque_val) const;
    bool isPrivate(::mlir::Operation *tablegen_opaque_val) const;
    bool isPublic(::mlir::Operation *tablegen_opaque_val) const;
    void setVisibility(::mlir::Operation *tablegen_opaque_val, mlir::SymbolTable::Visibility vis) const;
    void setNested(::mlir::Operation *tablegen_opaque_val) const;
    void setPrivate(::mlir::Operation *tablegen_opaque_val) const;
    void setPublic(::mlir::Operation *tablegen_opaque_val) const;
    Optional<::mlir::SymbolTable::UseRange> getSymbolUses(::mlir::Operation *tablegen_opaque_val, Operation *from) const;
    bool symbolKnownUseEmpty(::mlir::Operation *tablegen_opaque_val, Operation *from) const;
    LogicalResult replaceAllSymbolUses(::mlir::Operation *tablegen_opaque_val, StringAttr newSymbol, Operation *from) const;
    bool isOptionalSymbol(::mlir::Operation *tablegen_opaque_val) const;
    bool canDiscardOnUseEmpty(::mlir::Operation *tablegen_opaque_val) const;
    bool isDeclaration(::mlir::Operation *tablegen_opaque_val) const;
  };
};template <typename ConcreteOp>
struct SymbolOpInterfaceTrait;

} // namespace detail
class SymbolOpInterface : public ::mlir::OpInterface<SymbolOpInterface, detail::SymbolOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<SymbolOpInterface, detail::SymbolOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::SymbolOpInterfaceTrait<ConcreteOp> {};
  StringAttr getNameAttr();
  void setName(StringAttr name);
  mlir::SymbolTable::Visibility getVisibility();
  bool isNested();
  bool isPrivate();
  bool isPublic();
  void setVisibility(mlir::SymbolTable::Visibility vis);
  void setNested();
  void setPrivate();
  void setPublic();
  Optional<::mlir::SymbolTable::UseRange> getSymbolUses(Operation * from);
  bool symbolKnownUseEmpty(Operation * from);
  LogicalResult replaceAllSymbolUses(StringAttr newSymbol, Operation * from);
  bool isOptionalSymbol();
  bool canDiscardOnUseEmpty();
  bool isDeclaration();

    /// Convenience version of `getNameAttr` that returns a StringRef.
    StringRef getName() {
      return getNameAttr().getValue();
    }

    /// Convenience version of `setName` that take a StringRef.
    void setName(StringRef name) {
      setName(StringAttr::get(this->getContext(), name));
    }

    /// Custom classof that handles the case where the symbol is optional.
    static bool classof(Operation *op) {
      auto *opConcept = getInterfaceFor(op);
      if (!opConcept)
        return false;
      return !opConcept->isOptionalSymbol(opConcept, op) ||
             op->getAttr(::mlir::SymbolTable::getSymbolAttrName());
    }
  
};
namespace detail {
  template <typename ConcreteOp>
  struct SymbolOpInterfaceTrait : public ::mlir::OpInterface<SymbolOpInterface, detail::SymbolOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    StringAttr getNameAttr() {
      return mlir::SymbolTable::getSymbolName(this->getOperation());
    }
    void setName(StringAttr name) {
      this->getOperation()->setAttr(
            mlir::SymbolTable::getSymbolAttrName(), name);
    }
    mlir::SymbolTable::Visibility getVisibility() {
      return mlir::SymbolTable::getSymbolVisibility(this->getOperation());
    }
    bool isNested() {
      return getVisibility() == mlir::SymbolTable::Visibility::Nested;
    }
    bool isPrivate() {
      return getVisibility() == mlir::SymbolTable::Visibility::Private;
    }
    bool isPublic() {
      return getVisibility() == mlir::SymbolTable::Visibility::Public;
    }
    void setVisibility(mlir::SymbolTable::Visibility vis) {
      mlir::SymbolTable::setSymbolVisibility(this->getOperation(), vis);
    }
    void setNested() {
      setVisibility(mlir::SymbolTable::Visibility::Nested);
    }
    void setPrivate() {
      setVisibility(mlir::SymbolTable::Visibility::Private);
    }
    void setPublic() {
      setVisibility(mlir::SymbolTable::Visibility::Public);
    }
    Optional<::mlir::SymbolTable::UseRange> getSymbolUses(Operation * from) {
      return ::mlir::SymbolTable::getSymbolUses(this->getOperation(), from);
    }
    bool symbolKnownUseEmpty(Operation * from) {
      return ::mlir::SymbolTable::symbolKnownUseEmpty(this->getOperation(),
                                                        from);
    }
    LogicalResult replaceAllSymbolUses(StringAttr newSymbol, Operation * from) {
      return ::mlir::SymbolTable::replaceAllSymbolUses(this->getOperation(),
                                                         newSymbol, from);
    }
    bool isOptionalSymbol() {
      return false;
    }
    bool canDiscardOnUseEmpty() {
      // By default, base this on the visibility alone. A symbol can be
        // discarded as long as it is not public. Only public symbols may be
        // visible from outside of the IR.
        return getVisibility() != ::mlir::SymbolTable::Visibility::Public;
    }
    bool isDeclaration() {
      // By default, assume that the operation defines a symbol.
        return false;
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      // If this is an optional symbol, bail out early if possible.
    auto concreteOp = cast<ConcreteOp>(op);
    if (concreteOp.isOptionalSymbol()) {
      if(!concreteOp->getAttr(::mlir::SymbolTable::getSymbolAttrName()))
        return success();
    }
    if (::mlir::failed(::mlir::detail::verifySymbol(op)))
      return ::mlir::failure();
    if (concreteOp.isDeclaration() && concreteOp.isPublic())
      return concreteOp.emitOpError("symbol declaration cannot have public "
             "visibility");
    return success();
    }

    using Visibility = mlir::SymbolTable::Visibility;

    /// Convenience version of `getNameAttr` that returns a StringRef.
    StringRef getName() {
      return getNameAttr().getValue();
    }

    /// Convenience version of `setName` that take a StringRef.
    void setName(StringRef name) {
      setName(StringAttr::get((*static_cast<ConcreteOp *>(this))->getContext(), name));
    }
  
  };
}// namespace detail
template<typename ConcreteOp>
StringAttr detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::getNameAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  // Don't rely on the trait implementation as optional symbol operations
        // may override this.
        return mlir::SymbolTable::getSymbolName((llvm::cast<ConcreteOp>(tablegen_opaque_val)));
}
template<typename ConcreteOp>
void detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::setName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, StringAttr name) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setName(name);
}
template<typename ConcreteOp>
mlir::SymbolTable::Visibility detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::getVisibility(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getVisibility();
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::isNested(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isNested();
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::isPrivate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isPrivate();
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::isPublic(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isPublic();
}
template<typename ConcreteOp>
void detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::setVisibility(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::SymbolTable::Visibility vis) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setVisibility(vis);
}
template<typename ConcreteOp>
void detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::setNested(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setNested();
}
template<typename ConcreteOp>
void detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::setPrivate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setPrivate();
}
template<typename ConcreteOp>
void detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::setPublic(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setPublic();
}
template<typename ConcreteOp>
Optional<::mlir::SymbolTable::UseRange> detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSymbolUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Operation * from) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSymbolUses(from);
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::symbolKnownUseEmpty(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Operation * from) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).symbolKnownUseEmpty(from);
}
template<typename ConcreteOp>
LogicalResult detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::replaceAllSymbolUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, StringAttr newSymbol, Operation * from) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).replaceAllSymbolUses(newSymbol, from);
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::isOptionalSymbol(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isOptionalSymbol();
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::canDiscardOnUseEmpty(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).canDiscardOnUseEmpty();
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::Model<ConcreteOp>::isDeclaration(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDeclaration();
}
template<typename ConcreteOp>
StringAttr detail::SymbolOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getNameAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getNameAttr(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::SymbolOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, StringAttr name) {
  return static_cast<const ConcreteOp *>(impl)->setName(tablegen_opaque_val, name);
}
template<typename ConcreteOp>
mlir::SymbolTable::Visibility detail::SymbolOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getVisibility(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getVisibility(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isNested(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isNested(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isPrivate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isPrivate(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isPublic(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isPublic(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::SymbolOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setVisibility(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::SymbolTable::Visibility vis) {
  return static_cast<const ConcreteOp *>(impl)->setVisibility(tablegen_opaque_val, vis);
}
template<typename ConcreteOp>
void detail::SymbolOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setNested(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->setNested(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::SymbolOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setPrivate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->setPrivate(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::SymbolOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setPublic(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->setPublic(tablegen_opaque_val);
}
template<typename ConcreteOp>
Optional<::mlir::SymbolTable::UseRange> detail::SymbolOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSymbolUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Operation * from) {
  return static_cast<const ConcreteOp *>(impl)->getSymbolUses(tablegen_opaque_val, from);
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::symbolKnownUseEmpty(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Operation * from) {
  return static_cast<const ConcreteOp *>(impl)->symbolKnownUseEmpty(tablegen_opaque_val, from);
}
template<typename ConcreteOp>
LogicalResult detail::SymbolOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::replaceAllSymbolUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, StringAttr newSymbol, Operation * from) {
  return static_cast<const ConcreteOp *>(impl)->replaceAllSymbolUses(tablegen_opaque_val, newSymbol, from);
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isOptionalSymbol(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isOptionalSymbol(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::canDiscardOnUseEmpty(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->canDiscardOnUseEmpty(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isDeclaration(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isDeclaration(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
StringAttr detail::SymbolOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getNameAttr(::mlir::Operation *tablegen_opaque_val) const {
return mlir::SymbolTable::getSymbolName(this->getOperation());
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::SymbolOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setName(::mlir::Operation *tablegen_opaque_val, StringAttr name) const {
this->getOperation()->setAttr(
            mlir::SymbolTable::getSymbolAttrName(), name);
}
template<typename ConcreteModel, typename ConcreteOp>
mlir::SymbolTable::Visibility detail::SymbolOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getVisibility(::mlir::Operation *tablegen_opaque_val) const {
return mlir::SymbolTable::getSymbolVisibility(this->getOperation());
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isNested(::mlir::Operation *tablegen_opaque_val) const {
return getVisibility() == mlir::SymbolTable::Visibility::Nested;
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isPrivate(::mlir::Operation *tablegen_opaque_val) const {
return getVisibility() == mlir::SymbolTable::Visibility::Private;
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isPublic(::mlir::Operation *tablegen_opaque_val) const {
return getVisibility() == mlir::SymbolTable::Visibility::Public;
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::SymbolOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setVisibility(::mlir::Operation *tablegen_opaque_val, mlir::SymbolTable::Visibility vis) const {
mlir::SymbolTable::setSymbolVisibility(this->getOperation(), vis);
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::SymbolOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setNested(::mlir::Operation *tablegen_opaque_val) const {
setVisibility(mlir::SymbolTable::Visibility::Nested);
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::SymbolOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setPrivate(::mlir::Operation *tablegen_opaque_val) const {
setVisibility(mlir::SymbolTable::Visibility::Private);
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::SymbolOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setPublic(::mlir::Operation *tablegen_opaque_val) const {
setVisibility(mlir::SymbolTable::Visibility::Public);
}
template<typename ConcreteModel, typename ConcreteOp>
Optional<::mlir::SymbolTable::UseRange> detail::SymbolOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getSymbolUses(::mlir::Operation *tablegen_opaque_val, Operation *from) const {
return ::mlir::SymbolTable::getSymbolUses(this->getOperation(), from);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::symbolKnownUseEmpty(::mlir::Operation *tablegen_opaque_val, Operation *from) const {
return ::mlir::SymbolTable::symbolKnownUseEmpty(this->getOperation(),
                                                        from);
}
template<typename ConcreteModel, typename ConcreteOp>
LogicalResult detail::SymbolOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::replaceAllSymbolUses(::mlir::Operation *tablegen_opaque_val, StringAttr newSymbol, Operation *from) const {
return ::mlir::SymbolTable::replaceAllSymbolUses(this->getOperation(),
                                                         newSymbol, from);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isOptionalSymbol(::mlir::Operation *tablegen_opaque_val) const {
return false;
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::canDiscardOnUseEmpty(::mlir::Operation *tablegen_opaque_val) const {
// By default, base this on the visibility alone. A symbol can be
        // discarded as long as it is not public. Only public symbols may be
        // visible from outside of the IR.
        return getVisibility() != ::mlir::SymbolTable::Visibility::Public;
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::SymbolOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isDeclaration(::mlir::Operation *tablegen_opaque_val) const {
// By default, assume that the operation defines a symbol.
        return false;
}
} // namespace mlir
namespace mlir {
class SymbolUserOpInterface;
namespace detail {
struct SymbolUserOpInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::LogicalResult (*verifySymbolUses)(const Concept *impl, ::mlir::Operation *, ::mlir::SymbolTableCollection &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::SymbolUserOpInterface;
    Model() : Concept{verifySymbolUses} {}

    static inline ::mlir::LogicalResult verifySymbolUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::SymbolTableCollection & symbolTable);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::SymbolUserOpInterface;
    FallbackModel() : Concept{verifySymbolUses} {}

    static inline ::mlir::LogicalResult verifySymbolUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::SymbolTableCollection & symbolTable);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
  };
};template <typename ConcreteOp>
struct SymbolUserOpInterfaceTrait;

} // namespace detail
class SymbolUserOpInterface : public ::mlir::OpInterface<SymbolUserOpInterface, detail::SymbolUserOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<SymbolUserOpInterface, detail::SymbolUserOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::SymbolUserOpInterfaceTrait<ConcreteOp> {};
  ::mlir::LogicalResult verifySymbolUses(::mlir::SymbolTableCollection & symbolTable);
};
namespace detail {
  template <typename ConcreteOp>
  struct SymbolUserOpInterfaceTrait : public ::mlir::OpInterface<SymbolUserOpInterface, detail::SymbolUserOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
template<typename ConcreteOp>
::mlir::LogicalResult detail::SymbolUserOpInterfaceInterfaceTraits::Model<ConcreteOp>::verifySymbolUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::SymbolTableCollection & symbolTable) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).verifySymbolUses(symbolTable);
}
template<typename ConcreteOp>
::mlir::LogicalResult detail::SymbolUserOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::verifySymbolUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::SymbolTableCollection & symbolTable) {
  return static_cast<const ConcreteOp *>(impl)->verifySymbolUses(tablegen_opaque_val, symbolTable);
}
} // namespace mlir
