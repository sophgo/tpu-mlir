/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class PromotableAllocationOpInterface;
namespace detail {
struct PromotableAllocationOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::llvm::SmallVector<::mlir::MemorySlot> (*getPromotableSlots)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getDefaultValue)(const Concept *impl, ::mlir::Operation *, const ::mlir::MemorySlot &, ::mlir::OpBuilder &);
    void (*handleBlockArgument)(const Concept *impl, ::mlir::Operation *, const ::mlir::MemorySlot &, ::mlir::BlockArgument, ::mlir::OpBuilder &);
    void (*handlePromotionComplete)(const Concept *impl, ::mlir::Operation *, const ::mlir::MemorySlot &, ::mlir::Value);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::PromotableAllocationOpInterface;
    Model() : Concept{getPromotableSlots, getDefaultValue, handleBlockArgument, handlePromotionComplete} {}

    static inline ::llvm::SmallVector<::mlir::MemorySlot> getPromotableSlots(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getDefaultValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::OpBuilder & builder);
    static inline void handleBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::BlockArgument argument, ::mlir::OpBuilder & builder);
    static inline void handlePromotionComplete(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::Value defaultValue);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::PromotableAllocationOpInterface;
    FallbackModel() : Concept{getPromotableSlots, getDefaultValue, handleBlockArgument, handlePromotionComplete} {}

    static inline ::llvm::SmallVector<::mlir::MemorySlot> getPromotableSlots(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getDefaultValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::OpBuilder & builder);
    static inline void handleBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::BlockArgument argument, ::mlir::OpBuilder & builder);
    static inline void handlePromotionComplete(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::Value defaultValue);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};template <typename ConcreteOp>
struct PromotableAllocationOpInterfaceTrait;

} // namespace detail
class PromotableAllocationOpInterface : public ::mlir::OpInterface<PromotableAllocationOpInterface, detail::PromotableAllocationOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<PromotableAllocationOpInterface, detail::PromotableAllocationOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::PromotableAllocationOpInterfaceTrait<ConcreteOp> {};
  /// Returns a list of memory slots for which promotion should be attempted.
  /// This only considers the local semantics of the allocator, ignoring
  /// whether the slot pointer is properly used or not. This allocator is the
  /// "owner" of the returned slots, meaning no two allocators should return
  /// the same slot. The content of the memory slot must only be reachable
  /// using loads and stores to the provided slot pointer, no aliasing is
  /// allowed.
  /// 
  /// Promotion of the slot will lead to the slot pointer no longer being
  /// used, leaving the content of the memory slot unreachable.
  ::llvm::SmallVector<::mlir::MemorySlot> getPromotableSlots();
  /// Provides the default Value of this memory slot. The provided Value
  /// will be used as the reaching definition of loads done before any store.
  /// This Value must outlive the promotion and dominate all the uses of this
  /// slot's pointer. The provided builder can be used to create the default
  /// value on the fly.
  /// 
  /// The builder is located at the beginning of the block where the slot
  /// pointer is defined.
  ::mlir::Value getDefaultValue(const ::mlir::MemorySlot & slot, ::mlir::OpBuilder & builder);
  /// Hook triggered for every new block argument added to a block.
  /// This will only be called for slots declared by this operation.
  /// 
  /// The builder is located at the beginning of the block on call.
  void handleBlockArgument(const ::mlir::MemorySlot & slot, ::mlir::BlockArgument argument, ::mlir::OpBuilder & builder);
  /// Hook triggered once the promotion of a slot is complete. This can
  /// also clean up the created default value if necessary.
  /// This will only be called for slots declared by this operation.
  void handlePromotionComplete(const ::mlir::MemorySlot & slot, ::mlir::Value defaultValue);
};
namespace detail {
  template <typename ConcreteOp>
  struct PromotableAllocationOpInterfaceTrait : public ::mlir::OpInterface<PromotableAllocationOpInterface, detail::PromotableAllocationOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class PromotableMemOpInterface;
namespace detail {
struct PromotableMemOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    bool (*loadsFrom)(const Concept *impl, ::mlir::Operation *, const ::mlir::MemorySlot &);
    ::mlir::Value (*getStored)(const Concept *impl, ::mlir::Operation *, const ::mlir::MemorySlot &);
    bool (*canUsesBeRemoved)(const Concept *impl, ::mlir::Operation *, const ::mlir::MemorySlot &, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> &, ::llvm::SmallVectorImpl<::mlir::OpOperand *> &);
    ::mlir::DeletionKind (*removeBlockingUses)(const Concept *impl, ::mlir::Operation *, const ::mlir::MemorySlot &, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> &, ::mlir::OpBuilder &, ::mlir::Value);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::PromotableMemOpInterface;
    Model() : Concept{loadsFrom, getStored, canUsesBeRemoved, removeBlockingUses} {}

    static inline bool loadsFrom(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot);
    static inline ::mlir::Value getStored(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot);
    static inline bool canUsesBeRemoved(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> & blockingUses, ::llvm::SmallVectorImpl<::mlir::OpOperand *> & newBlockingUses);
    static inline ::mlir::DeletionKind removeBlockingUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> & blockingUses, ::mlir::OpBuilder & builder, ::mlir::Value reachingDefinition);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::PromotableMemOpInterface;
    FallbackModel() : Concept{loadsFrom, getStored, canUsesBeRemoved, removeBlockingUses} {}

    static inline bool loadsFrom(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot);
    static inline ::mlir::Value getStored(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot);
    static inline bool canUsesBeRemoved(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> & blockingUses, ::llvm::SmallVectorImpl<::mlir::OpOperand *> & newBlockingUses);
    static inline ::mlir::DeletionKind removeBlockingUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> & blockingUses, ::mlir::OpBuilder & builder, ::mlir::Value reachingDefinition);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};template <typename ConcreteOp>
struct PromotableMemOpInterfaceTrait;

} // namespace detail
class PromotableMemOpInterface : public ::mlir::OpInterface<PromotableMemOpInterface, detail::PromotableMemOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<PromotableMemOpInterface, detail::PromotableMemOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::PromotableMemOpInterfaceTrait<ConcreteOp> {};
  /// Gets whether this operation loads from the specified slot.
  bool loadsFrom(const ::mlir::MemorySlot & slot);
  /// Gets the value stored to the provided memory slot, or returns a null
  /// value if this operation does not store to this slot. An operation
  /// storing a value to a slot must always be able to provide the value it
  /// stores. This method is only called on operations that use the slot.
  ::mlir::Value getStored(const ::mlir::MemorySlot & slot);
  /// Checks that this operation can be promoted to no longer use the provided
  /// blocking uses, in the context of promoting `slot`.
  /// 
  /// If the removal procedure of the use will require that other uses get
  /// removed, that dependency should be added to the `newBlockingUses`
  /// argument. Dependent uses must only be uses of results of this operation.
  bool canUsesBeRemoved(const ::mlir::MemorySlot & slot, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> & blockingUses, ::llvm::SmallVectorImpl<::mlir::OpOperand *> & newBlockingUses);
  /// Transforms IR to ensure that the current operation does not use the
  /// provided memory slot anymore. `reachingDefinition` contains the value
  /// currently stored in the provided memory slot, immediately before the
  /// current operation.
  /// 
  /// During the transformation, *no operation should be deleted*.
  /// The operation can only schedule its own deletion by returning the
  /// appropriate `DeletionKind`. The deletion must be legal assuming the
  /// blocking uses passed through the `newBlockingUses` list in
  /// `canUseBeRemoved` have been removed.
  /// 
  /// After calling this method, the blocking uses should have disappeared
  /// or this operation should have scheduled its own deletion.
  /// 
  /// This method will only be called after ensuring promotion is allowed via
  /// `canUseBeRemoved`. The requested blocking use removal may or may not
  /// have been done at the point of calling this method, but it will be done
  /// eventually.
  /// 
  /// The builder is located after the promotable operation on call.
  ::mlir::DeletionKind removeBlockingUses(const ::mlir::MemorySlot & slot, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> & blockingUses, ::mlir::OpBuilder & builder, ::mlir::Value reachingDefinition);
};
namespace detail {
  template <typename ConcreteOp>
  struct PromotableMemOpInterfaceTrait : public ::mlir::OpInterface<PromotableMemOpInterface, detail::PromotableMemOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class PromotableOpInterface;
namespace detail {
struct PromotableOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    bool (*canUsesBeRemoved)(const Concept *impl, ::mlir::Operation *, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> &, ::llvm::SmallVectorImpl<::mlir::OpOperand *> &);
    ::mlir::DeletionKind (*removeBlockingUses)(const Concept *impl, ::mlir::Operation *, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> &, ::mlir::OpBuilder &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::PromotableOpInterface;
    Model() : Concept{canUsesBeRemoved, removeBlockingUses} {}

    static inline bool canUsesBeRemoved(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> & blockingUses, ::llvm::SmallVectorImpl<::mlir::OpOperand *> & newBlockingUses);
    static inline ::mlir::DeletionKind removeBlockingUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> & blockingUses, ::mlir::OpBuilder & builder);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::PromotableOpInterface;
    FallbackModel() : Concept{canUsesBeRemoved, removeBlockingUses} {}

    static inline bool canUsesBeRemoved(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> & blockingUses, ::llvm::SmallVectorImpl<::mlir::OpOperand *> & newBlockingUses);
    static inline ::mlir::DeletionKind removeBlockingUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> & blockingUses, ::mlir::OpBuilder & builder);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};template <typename ConcreteOp>
struct PromotableOpInterfaceTrait;

} // namespace detail
class PromotableOpInterface : public ::mlir::OpInterface<PromotableOpInterface, detail::PromotableOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<PromotableOpInterface, detail::PromotableOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::PromotableOpInterfaceTrait<ConcreteOp> {};
  /// Checks that this operation can be promoted to no longer use the provided
  /// blocking uses, in order to allow optimization.
  /// 
  /// If the removal procedure of the use will require that other uses get
  /// removed, that dependency should be added to the `newBlockingUses`
  /// argument. Dependent uses must only be uses of results of this operation.
  bool canUsesBeRemoved(const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> & blockingUses, ::llvm::SmallVectorImpl<::mlir::OpOperand *> & newBlockingUses);
  /// Transforms IR to ensure that the current operation does not use the
  /// provided blocking uses anymore. In contrast to
  /// `PromotableMemOpInterface`, operations implementing this interface
  /// must not need access to the reaching definition of the content of the
  /// slot.
  /// 
  /// During the transformation, *no operation should be deleted*.
  /// The operation can only schedule its own deletion by returning the
  /// appropriate `DeletionKind`. The deletion must be legal assuming the
  /// blocking uses passed through the `newBlockingUses` list in
  /// `canUseBeRemoved` have been removed.
  /// 
  /// After calling this method, the blocking uses should have disappeared
  /// or this operation should have scheduled its own deletion.
  /// 
  /// This method will only be called after ensuring promotion is allowed via
  /// `canUseBeRemoved`. The requested blocking use removal may or may not
  /// have been done at the point of calling this method, but it will be done
  /// eventually.
  /// 
  /// The builder is located after the promotable operation on call.
  ::mlir::DeletionKind removeBlockingUses(const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> & blockingUses, ::mlir::OpBuilder & builder);
};
namespace detail {
  template <typename ConcreteOp>
  struct PromotableOpInterfaceTrait : public ::mlir::OpInterface<PromotableOpInterface, detail::PromotableOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::MemorySlot> detail::PromotableAllocationOpInterfaceInterfaceTraits::Model<ConcreteOp>::getPromotableSlots(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getPromotableSlots();
}
template<typename ConcreteOp>
::mlir::Value detail::PromotableAllocationOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDefaultValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::OpBuilder & builder) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDefaultValue(slot, builder);
}
template<typename ConcreteOp>
void detail::PromotableAllocationOpInterfaceInterfaceTraits::Model<ConcreteOp>::handleBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::BlockArgument argument, ::mlir::OpBuilder & builder) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).handleBlockArgument(slot, argument, builder);
}
template<typename ConcreteOp>
void detail::PromotableAllocationOpInterfaceInterfaceTraits::Model<ConcreteOp>::handlePromotionComplete(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::Value defaultValue) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).handlePromotionComplete(slot, defaultValue);
}
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::MemorySlot> detail::PromotableAllocationOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getPromotableSlots(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getPromotableSlots(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::PromotableAllocationOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDefaultValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::OpBuilder & builder) {
  return static_cast<const ConcreteOp *>(impl)->getDefaultValue(tablegen_opaque_val, slot, builder);
}
template<typename ConcreteOp>
void detail::PromotableAllocationOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::handleBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::BlockArgument argument, ::mlir::OpBuilder & builder) {
  return static_cast<const ConcreteOp *>(impl)->handleBlockArgument(tablegen_opaque_val, slot, argument, builder);
}
template<typename ConcreteOp>
void detail::PromotableAllocationOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::handlePromotionComplete(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, ::mlir::Value defaultValue) {
  return static_cast<const ConcreteOp *>(impl)->handlePromotionComplete(tablegen_opaque_val, slot, defaultValue);
}
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
bool detail::PromotableMemOpInterfaceInterfaceTraits::Model<ConcreteOp>::loadsFrom(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).loadsFrom(slot);
}
template<typename ConcreteOp>
::mlir::Value detail::PromotableMemOpInterfaceInterfaceTraits::Model<ConcreteOp>::getStored(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStored(slot);
}
template<typename ConcreteOp>
bool detail::PromotableMemOpInterfaceInterfaceTraits::Model<ConcreteOp>::canUsesBeRemoved(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> & blockingUses, ::llvm::SmallVectorImpl<::mlir::OpOperand *> & newBlockingUses) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).canUsesBeRemoved(slot, blockingUses, newBlockingUses);
}
template<typename ConcreteOp>
::mlir::DeletionKind detail::PromotableMemOpInterfaceInterfaceTraits::Model<ConcreteOp>::removeBlockingUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> & blockingUses, ::mlir::OpBuilder & builder, ::mlir::Value reachingDefinition) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).removeBlockingUses(slot, blockingUses, builder, reachingDefinition);
}
template<typename ConcreteOp>
bool detail::PromotableMemOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::loadsFrom(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot) {
  return static_cast<const ConcreteOp *>(impl)->loadsFrom(tablegen_opaque_val, slot);
}
template<typename ConcreteOp>
::mlir::Value detail::PromotableMemOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getStored(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot) {
  return static_cast<const ConcreteOp *>(impl)->getStored(tablegen_opaque_val, slot);
}
template<typename ConcreteOp>
bool detail::PromotableMemOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::canUsesBeRemoved(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> & blockingUses, ::llvm::SmallVectorImpl<::mlir::OpOperand *> & newBlockingUses) {
  return static_cast<const ConcreteOp *>(impl)->canUsesBeRemoved(tablegen_opaque_val, slot, blockingUses, newBlockingUses);
}
template<typename ConcreteOp>
::mlir::DeletionKind detail::PromotableMemOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::removeBlockingUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::MemorySlot & slot, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> & blockingUses, ::mlir::OpBuilder & builder, ::mlir::Value reachingDefinition) {
  return static_cast<const ConcreteOp *>(impl)->removeBlockingUses(tablegen_opaque_val, slot, blockingUses, builder, reachingDefinition);
}
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
bool detail::PromotableOpInterfaceInterfaceTraits::Model<ConcreteOp>::canUsesBeRemoved(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> & blockingUses, ::llvm::SmallVectorImpl<::mlir::OpOperand *> & newBlockingUses) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).canUsesBeRemoved(blockingUses, newBlockingUses);
}
template<typename ConcreteOp>
::mlir::DeletionKind detail::PromotableOpInterfaceInterfaceTraits::Model<ConcreteOp>::removeBlockingUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> & blockingUses, ::mlir::OpBuilder & builder) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).removeBlockingUses(blockingUses, builder);
}
template<typename ConcreteOp>
bool detail::PromotableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::canUsesBeRemoved(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> & blockingUses, ::llvm::SmallVectorImpl<::mlir::OpOperand *> & newBlockingUses) {
  return static_cast<const ConcreteOp *>(impl)->canUsesBeRemoved(tablegen_opaque_val, blockingUses, newBlockingUses);
}
template<typename ConcreteOp>
::mlir::DeletionKind detail::PromotableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::removeBlockingUses(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> & blockingUses, ::mlir::OpBuilder & builder) {
  return static_cast<const ConcreteOp *>(impl)->removeBlockingUses(tablegen_opaque_val, blockingUses, builder);
}
} // namespace mlir
