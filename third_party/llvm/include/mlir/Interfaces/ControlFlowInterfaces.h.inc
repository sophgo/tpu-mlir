/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class BranchOpInterface;
namespace detail {
struct BranchOpInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::Optional<::mlir::MutableOperandRange> (*getMutableSuccessorOperands)(const Concept *impl, ::mlir::Operation *, unsigned);
    ::mlir::Optional<::mlir::OperandRange> (*getSuccessorOperands)(const Concept *impl, ::mlir::Operation *, unsigned);
    ::mlir::Optional<::mlir::BlockArgument> (*getSuccessorBlockArgument)(const Concept *impl, ::mlir::Operation *, unsigned);
    ::mlir::Block *(*getSuccessorForOperands)(const Concept *impl, ::mlir::Operation *, ::mlir::ArrayRef<::mlir::Attribute>);
    bool (*areTypesCompatible)(const Concept *impl, ::mlir::Operation *, ::mlir::Type, ::mlir::Type);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::BranchOpInterface;
    Model() : Concept{getMutableSuccessorOperands, getSuccessorOperands, getSuccessorBlockArgument, getSuccessorForOperands, areTypesCompatible} {}

    static inline ::mlir::Optional<::mlir::MutableOperandRange> getMutableSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index);
    static inline ::mlir::Optional<::mlir::OperandRange> getSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index);
    static inline ::mlir::Optional<::mlir::BlockArgument> getSuccessorBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned operandIndex);
    static inline ::mlir::Block *getSuccessorForOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayRef<::mlir::Attribute> operands);
    static inline bool areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::BranchOpInterface;
    FallbackModel() : Concept{getMutableSuccessorOperands, getSuccessorOperands, getSuccessorBlockArgument, getSuccessorForOperands, areTypesCompatible} {}

    static inline ::mlir::Optional<::mlir::MutableOperandRange> getMutableSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index);
    static inline ::mlir::Optional<::mlir::OperandRange> getSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index);
    static inline ::mlir::Optional<::mlir::BlockArgument> getSuccessorBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned operandIndex);
    static inline ::mlir::Block *getSuccessorForOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayRef<::mlir::Attribute> operands);
    static inline bool areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    ::mlir::Optional<::mlir::OperandRange> getSuccessorOperands(::mlir::Operation *tablegen_opaque_val, unsigned index) const;
    ::mlir::Block *getSuccessorForOperands(::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayRef<::mlir::Attribute> operands) const;
    bool areTypesCompatible(::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) const;
  };
};template <typename ConcreteOp>
struct BranchOpInterfaceTrait;

} // namespace detail
class BranchOpInterface : public ::mlir::OpInterface<BranchOpInterface, detail::BranchOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<BranchOpInterface, detail::BranchOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::BranchOpInterfaceTrait<ConcreteOp> {};
  ::mlir::Optional<::mlir::MutableOperandRange> getMutableSuccessorOperands(unsigned index);
  ::mlir::Optional<::mlir::OperandRange> getSuccessorOperands(unsigned index);
  ::mlir::Optional<::mlir::BlockArgument> getSuccessorBlockArgument(unsigned operandIndex);
  ::mlir::Block *getSuccessorForOperands(::mlir::ArrayRef<::mlir::Attribute> operands);
  bool areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs);
};
namespace detail {
  template <typename ConcreteOp>
  struct BranchOpInterfaceTrait : public ::mlir::OpInterface<BranchOpInterface, detail::BranchOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    ::mlir::Optional<::mlir::OperandRange> getSuccessorOperands(unsigned index) {
      auto operands = (*static_cast<ConcreteOp *>(this)).getMutableSuccessorOperands(index);
        return operands ? ::mlir::Optional<::mlir::OperandRange>(*operands) : ::llvm::None;
    }
    ::mlir::Block *getSuccessorForOperands(::mlir::ArrayRef<::mlir::Attribute> operands) {
      return nullptr;
    }
    bool areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs) {
      return lhs == rhs;
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      auto concreteOp = ::mlir::cast<ConcreteOp>(op);
    for (unsigned i = 0, e = op->getNumSuccessors(); i != e; ++i) {
      ::mlir::Optional<OperandRange> operands = concreteOp.getSuccessorOperands(i);
      if (::mlir::failed(::mlir::detail::verifyBranchSuccessorOperands(op, i, operands)))
        return ::mlir::failure();
    }
    return ::mlir::success();
    }
  };
}// namespace detail
template<typename ConcreteOp>
::mlir::Optional<::mlir::MutableOperandRange> detail::BranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMutableSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMutableSuccessorOperands(index);
}
template<typename ConcreteOp>
::mlir::Optional<::mlir::OperandRange> detail::BranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorOperands(index);
}
template<typename ConcreteOp>
::mlir::Optional<::mlir::BlockArgument> detail::BranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned operandIndex) {
  ::mlir::Operation *opaqueOp = (llvm::cast<ConcreteOp>(tablegen_opaque_val));
        for (unsigned i = 0, e = opaqueOp->getNumSuccessors(); i != e; ++i) {
          if (::mlir::Optional<::mlir::BlockArgument> arg = ::mlir::detail::getBranchSuccessorArgument(
                (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorOperands(i), operandIndex,
                opaqueOp->getSuccessor(i)))
            return arg;
        }
        return ::llvm::None;
}
template<typename ConcreteOp>
::mlir::Block *detail::BranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorForOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayRef<::mlir::Attribute> operands) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorForOperands(operands);
}
template<typename ConcreteOp>
bool detail::BranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).areTypesCompatible(lhs, rhs);
}
template<typename ConcreteOp>
::mlir::Optional<::mlir::MutableOperandRange> detail::BranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMutableSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index) {
  return static_cast<const ConcreteOp *>(impl)->getMutableSuccessorOperands(tablegen_opaque_val, index);
}
template<typename ConcreteOp>
::mlir::Optional<::mlir::OperandRange> detail::BranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index) {
  return static_cast<const ConcreteOp *>(impl)->getSuccessorOperands(tablegen_opaque_val, index);
}
template<typename ConcreteOp>
::mlir::Optional<::mlir::BlockArgument> detail::BranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSuccessorBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned operandIndex) {
  return static_cast<const ConcreteOp *>(impl)->getSuccessorBlockArgument(tablegen_opaque_val, operandIndex);
}
template<typename ConcreteOp>
::mlir::Block *detail::BranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSuccessorForOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayRef<::mlir::Attribute> operands) {
  return static_cast<const ConcreteOp *>(impl)->getSuccessorForOperands(tablegen_opaque_val, operands);
}
template<typename ConcreteOp>
bool detail::BranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) {
  return static_cast<const ConcreteOp *>(impl)->areTypesCompatible(tablegen_opaque_val, lhs, rhs);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Optional<::mlir::OperandRange> detail::BranchOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getSuccessorOperands(::mlir::Operation *tablegen_opaque_val, unsigned index) const {
auto operands = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMutableSuccessorOperands(index);
        return operands ? ::mlir::Optional<::mlir::OperandRange>(*operands) : ::llvm::None;
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Block *detail::BranchOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getSuccessorForOperands(::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayRef<::mlir::Attribute> operands) const {
return nullptr;
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::BranchOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::areTypesCompatible(::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) const {
return lhs == rhs;
}
} // namespace mlir
namespace mlir {
class RegionBranchOpInterface;
namespace detail {
struct RegionBranchOpInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::OperandRange (*getSuccessorEntryOperands)(const Concept *impl, ::mlir::Operation *, unsigned);
    void (*getSuccessorRegions)(const Concept *impl, ::mlir::Operation *, ::mlir::Optional<unsigned>, ::mlir::ArrayRef<::mlir::Attribute>, ::mlir::SmallVectorImpl<::mlir::RegionSuccessor> &);
    void (*getRegionInvocationBounds)(const Concept *impl, ::mlir::Operation *, ::mlir::ArrayRef<::mlir::Attribute>, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> &);
    bool (*areTypesCompatible)(const Concept *impl, ::mlir::Operation *, ::mlir::Type, ::mlir::Type);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::RegionBranchOpInterface;
    Model() : Concept{getSuccessorEntryOperands, getSuccessorRegions, getRegionInvocationBounds, areTypesCompatible} {}

    static inline ::mlir::OperandRange getSuccessorEntryOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index);
    static inline void getSuccessorRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Optional<unsigned> index, ::mlir::ArrayRef<::mlir::Attribute> operands, ::mlir::SmallVectorImpl<::mlir::RegionSuccessor> & regions);
    static inline void getRegionInvocationBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds);
    static inline bool areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::RegionBranchOpInterface;
    FallbackModel() : Concept{getSuccessorEntryOperands, getSuccessorRegions, getRegionInvocationBounds, areTypesCompatible} {}

    static inline ::mlir::OperandRange getSuccessorEntryOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index);
    static inline void getSuccessorRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Optional<unsigned> index, ::mlir::ArrayRef<::mlir::Attribute> operands, ::mlir::SmallVectorImpl<::mlir::RegionSuccessor> & regions);
    static inline void getRegionInvocationBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds);
    static inline bool areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    ::mlir::OperandRange getSuccessorEntryOperands(::mlir::Operation *tablegen_opaque_val, unsigned index) const;
    void getRegionInvocationBounds(::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> &invocationBounds) const;
    bool areTypesCompatible(::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) const;
  };
};template <typename ConcreteOp>
struct RegionBranchOpInterfaceTrait;

} // namespace detail
class RegionBranchOpInterface : public ::mlir::OpInterface<RegionBranchOpInterface, detail::RegionBranchOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<RegionBranchOpInterface, detail::RegionBranchOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::RegionBranchOpInterfaceTrait<ConcreteOp> {};
  ::mlir::OperandRange getSuccessorEntryOperands(unsigned index);
  void getSuccessorRegions(::mlir::Optional<unsigned> index, ::mlir::ArrayRef<::mlir::Attribute> operands, ::mlir::SmallVectorImpl<::mlir::RegionSuccessor> & regions);
  void getRegionInvocationBounds(::mlir::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds);
  bool areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs);

    /// Convenience helper in case none of the operands is known.
    void getSuccessorRegions(Optional<unsigned> index,
                             SmallVectorImpl<RegionSuccessor> &regions) {
       SmallVector<Attribute, 2> nullAttrs(getOperation()->getNumOperands());
       getSuccessorRegions(index, nullAttrs, regions);
    }
  
};
namespace detail {
  template <typename ConcreteOp>
  struct RegionBranchOpInterfaceTrait : public ::mlir::OpInterface<RegionBranchOpInterface, detail::RegionBranchOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    ::mlir::OperandRange getSuccessorEntryOperands(unsigned index) {
      auto operandEnd = this->getOperation()->operand_end();
        return ::mlir::OperandRange(operandEnd, operandEnd);
    }
    void getRegionInvocationBounds(::mlir::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds) {
      invocationBounds.append((*static_cast<ConcreteOp *>(this))->getNumRegions(),
                                  ::mlir::InvocationBounds::getUnknown());
    }
    bool areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs) {
      return lhs == rhs;
    }
    static ::mlir::LogicalResult verifyRegionTrait(::mlir::Operation *op) {
      static_assert(!ConcreteOp::template hasTrait<OpTrait::ZeroRegion>(),
                  "expected operation to have non-zero regions");
    return detail::verifyTypesAlongControlFlowEdges(op);
    }
  };
}// namespace detail
template<typename ConcreteOp>
::mlir::OperandRange detail::RegionBranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorEntryOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorEntryOperands(index);
}
template<typename ConcreteOp>
void detail::RegionBranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Optional<unsigned> index, ::mlir::ArrayRef<::mlir::Attribute> operands, ::mlir::SmallVectorImpl<::mlir::RegionSuccessor> & regions) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorRegions(index, operands, regions);
}
template<typename ConcreteOp>
void detail::RegionBranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getRegionInvocationBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRegionInvocationBounds(operands, invocationBounds);
}
template<typename ConcreteOp>
bool detail::RegionBranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).areTypesCompatible(lhs, rhs);
}
template<typename ConcreteOp>
::mlir::OperandRange detail::RegionBranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSuccessorEntryOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index) {
  return static_cast<const ConcreteOp *>(impl)->getSuccessorEntryOperands(tablegen_opaque_val, index);
}
template<typename ConcreteOp>
void detail::RegionBranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSuccessorRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Optional<unsigned> index, ::mlir::ArrayRef<::mlir::Attribute> operands, ::mlir::SmallVectorImpl<::mlir::RegionSuccessor> & regions) {
  return static_cast<const ConcreteOp *>(impl)->getSuccessorRegions(tablegen_opaque_val, index, operands, regions);
}
template<typename ConcreteOp>
void detail::RegionBranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getRegionInvocationBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds) {
  return static_cast<const ConcreteOp *>(impl)->getRegionInvocationBounds(tablegen_opaque_val, operands, invocationBounds);
}
template<typename ConcreteOp>
bool detail::RegionBranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) {
  return static_cast<const ConcreteOp *>(impl)->areTypesCompatible(tablegen_opaque_val, lhs, rhs);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::OperandRange detail::RegionBranchOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getSuccessorEntryOperands(::mlir::Operation *tablegen_opaque_val, unsigned index) const {
auto operandEnd = this->getOperation()->operand_end();
        return ::mlir::OperandRange(operandEnd, operandEnd);
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::RegionBranchOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getRegionInvocationBounds(::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> &invocationBounds) const {
invocationBounds.append((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getNumRegions(),
                                  ::mlir::InvocationBounds::getUnknown());
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::RegionBranchOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::areTypesCompatible(::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) const {
return lhs == rhs;
}
} // namespace mlir
namespace mlir {
class RegionBranchTerminatorOpInterface;
namespace detail {
struct RegionBranchTerminatorOpInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::MutableOperandRange (*getMutableSuccessorOperands)(const Concept *impl, ::mlir::Operation *, ::mlir::Optional<unsigned>);
    ::mlir::OperandRange (*getSuccessorOperands)(const Concept *impl, ::mlir::Operation *, ::mlir::Optional<unsigned>);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::RegionBranchTerminatorOpInterface;
    Model() : Concept{getMutableSuccessorOperands, getSuccessorOperands} {}

    static inline ::mlir::MutableOperandRange getMutableSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Optional<unsigned> index);
    static inline ::mlir::OperandRange getSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Optional<unsigned> index);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::RegionBranchTerminatorOpInterface;
    FallbackModel() : Concept{getMutableSuccessorOperands, getSuccessorOperands} {}

    static inline ::mlir::MutableOperandRange getMutableSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Optional<unsigned> index);
    static inline ::mlir::OperandRange getSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Optional<unsigned> index);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    ::mlir::OperandRange getSuccessorOperands(::mlir::Operation *tablegen_opaque_val, ::mlir::Optional<unsigned> index) const;
  };
};template <typename ConcreteOp>
struct RegionBranchTerminatorOpInterfaceTrait;

} // namespace detail
class RegionBranchTerminatorOpInterface : public ::mlir::OpInterface<RegionBranchTerminatorOpInterface, detail::RegionBranchTerminatorOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<RegionBranchTerminatorOpInterface, detail::RegionBranchTerminatorOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::RegionBranchTerminatorOpInterfaceTrait<ConcreteOp> {};
  ::mlir::MutableOperandRange getMutableSuccessorOperands(::mlir::Optional<unsigned> index);
  ::mlir::OperandRange getSuccessorOperands(::mlir::Optional<unsigned> index);
};
namespace detail {
  template <typename ConcreteOp>
  struct RegionBranchTerminatorOpInterfaceTrait : public ::mlir::OpInterface<RegionBranchTerminatorOpInterface, detail::RegionBranchTerminatorOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    ::mlir::OperandRange getSuccessorOperands(::mlir::Optional<unsigned> index) {
      return (*static_cast<ConcreteOp *>(this)).getMutableSuccessorOperands(index);
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      static_assert(ConcreteOp::template hasTrait<OpTrait::IsTerminator>(),
                  "expected operation to be a terminator");
    static_assert(ConcreteOp::template hasTrait<OpTrait::ZeroResult>(),
                  "expected operation to have zero results");
    static_assert(ConcreteOp::template hasTrait<OpTrait::ZeroSuccessor>(),
                  "expected operation to have zero successors");
    return success();
    }
  };
}// namespace detail
template<typename ConcreteOp>
::mlir::MutableOperandRange detail::RegionBranchTerminatorOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMutableSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Optional<unsigned> index) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMutableSuccessorOperands(index);
}
template<typename ConcreteOp>
::mlir::OperandRange detail::RegionBranchTerminatorOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Optional<unsigned> index) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorOperands(index);
}
template<typename ConcreteOp>
::mlir::MutableOperandRange detail::RegionBranchTerminatorOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMutableSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Optional<unsigned> index) {
  return static_cast<const ConcreteOp *>(impl)->getMutableSuccessorOperands(tablegen_opaque_val, index);
}
template<typename ConcreteOp>
::mlir::OperandRange detail::RegionBranchTerminatorOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Optional<unsigned> index) {
  return static_cast<const ConcreteOp *>(impl)->getSuccessorOperands(tablegen_opaque_val, index);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::OperandRange detail::RegionBranchTerminatorOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getSuccessorOperands(::mlir::Operation *tablegen_opaque_val, ::mlir::Optional<unsigned> index) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMutableSuccessorOperands(index);
}
} // namespace mlir
