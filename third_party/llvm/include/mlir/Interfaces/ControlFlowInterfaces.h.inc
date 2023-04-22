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
    /// The methods defined by the interface.
    ::mlir::SuccessorOperands (*getSuccessorOperands)(const Concept *impl, ::mlir::Operation *, unsigned);
    ::std::optional<::mlir::BlockArgument> (*getSuccessorBlockArgument)(const Concept *impl, ::mlir::Operation *, unsigned);
    ::mlir::Block *(*getSuccessorForOperands)(const Concept *impl, ::mlir::Operation *, ::llvm::ArrayRef<::mlir::Attribute>);
    bool (*areTypesCompatible)(const Concept *impl, ::mlir::Operation *, ::mlir::Type, ::mlir::Type);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::BranchOpInterface;
    Model() : Concept{getSuccessorOperands, getSuccessorBlockArgument, getSuccessorForOperands, areTypesCompatible} {}

    static inline ::mlir::SuccessorOperands getSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index);
    static inline ::std::optional<::mlir::BlockArgument> getSuccessorBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned operandIndex);
    static inline ::mlir::Block *getSuccessorForOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands);
    static inline bool areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::BranchOpInterface;
    FallbackModel() : Concept{getSuccessorOperands, getSuccessorBlockArgument, getSuccessorForOperands, areTypesCompatible} {}

    static inline ::mlir::SuccessorOperands getSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index);
    static inline ::std::optional<::mlir::BlockArgument> getSuccessorBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned operandIndex);
    static inline ::mlir::Block *getSuccessorForOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands);
    static inline bool areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::Block *getSuccessorForOperands(::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands) const;
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
  /// Returns the operands that correspond to the arguments of the successor
  /// at the given index. It consists of a number of operands that are
  /// internally produced by the operation, followed by a range of operands
  /// that are forwarded. An example operation making use of produced
  /// operands would be:
  /// 
  /// ```mlir
  /// invoke %function(%0)
  ///     label ^success ^error(%1 : i32)
  /// 
  /// ^error(%e: !error, %arg0: i32):
  ///     ...
  /// ```
  /// 
  /// The operand that would map to the `^error`s `%e` operand is produced
  /// by the `invoke` operation, while `%1` is a forwarded operand that maps
  /// to `%arg0` in the successor.
  /// 
  /// Produced operands always map to the first few block arguments of the
  /// successor, followed by the forwarded operands. Mapping them in any
  /// other order is not supported by the interface.
  /// 
  /// By having the forwarded operands last allows users of the interface
  /// to append more forwarded operands to the branch operation without
  /// interfering with other successor operands.
  ::mlir::SuccessorOperands getSuccessorOperands(unsigned index);
  /// Returns the `BlockArgument` corresponding to operand `operandIndex` in
  /// some successor, or None if `operandIndex` isn't a successor operand
  /// index.
  ::std::optional<::mlir::BlockArgument> getSuccessorBlockArgument(unsigned operandIndex);
  /// Returns the successor that would be chosen with the given constant
  /// operands. Returns nullptr if a single successor could not be chosen.
  ::mlir::Block *getSuccessorForOperands(::llvm::ArrayRef<::mlir::Attribute> operands);
  /// This method is called to compare types along control-flow edges. By
  /// default, the types are checked as equal.
  bool areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs);
};
namespace detail {
  template <typename ConcreteOp>
  struct BranchOpInterfaceTrait : public ::mlir::OpInterface<BranchOpInterface, detail::BranchOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns the successor that would be chosen with the given constant
    /// operands. Returns nullptr if a single successor could not be chosen.
    ::mlir::Block *getSuccessorForOperands(::llvm::ArrayRef<::mlir::Attribute> operands) {
      return nullptr;
    }
    /// This method is called to compare types along control-flow edges. By
    /// default, the types are checked as equal.
    bool areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs) {
      return lhs == rhs;
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      auto concreteOp = ::mlir::cast<ConcreteOp>(op);
    for (unsigned i = 0, e = op->getNumSuccessors(); i != e; ++i) {
      ::mlir::SuccessorOperands operands = concreteOp.getSuccessorOperands(i);
      if (::mlir::failed(::mlir::detail::verifyBranchSuccessorOperands(op, i, operands)))
        return ::mlir::failure();
    }
    return ::mlir::success();
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class RegionBranchOpInterface;
namespace detail {
struct RegionBranchOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::OperandRange (*getSuccessorEntryOperands)(const Concept *impl, ::mlir::Operation *, ::std::optional<unsigned>);
    void (*getSuccessorRegions)(const Concept *impl, ::mlir::Operation *, ::std::optional<unsigned>, ::llvm::ArrayRef<::mlir::Attribute>, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &);
    void (*getRegionInvocationBounds)(const Concept *impl, ::mlir::Operation *, ::llvm::ArrayRef<::mlir::Attribute>, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> &);
    bool (*areTypesCompatible)(const Concept *impl, ::mlir::Operation *, ::mlir::Type, ::mlir::Type);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::RegionBranchOpInterface;
    Model() : Concept{getSuccessorEntryOperands, getSuccessorRegions, getRegionInvocationBounds, areTypesCompatible} {}

    static inline ::mlir::OperandRange getSuccessorEntryOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::std::optional<unsigned> index);
    static inline void getSuccessorRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::std::optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions);
    static inline void getRegionInvocationBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds);
    static inline bool areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::RegionBranchOpInterface;
    FallbackModel() : Concept{getSuccessorEntryOperands, getSuccessorRegions, getRegionInvocationBounds, areTypesCompatible} {}

    static inline ::mlir::OperandRange getSuccessorEntryOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::std::optional<unsigned> index);
    static inline void getSuccessorRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::std::optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions);
    static inline void getRegionInvocationBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds);
    static inline bool areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::OperandRange getSuccessorEntryOperands(::mlir::Operation *tablegen_opaque_val, ::std::optional<unsigned> index) const;
    void getRegionInvocationBounds(::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> &invocationBounds) const;
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
  /// Returns the operands of this operation used as the entry arguments when
  /// entering the region at `index`, which was specified as a successor of
  /// this operation by `getSuccessorRegions`, or the operands forwarded to
  /// the operation's results when it branches back to itself. These operands
  /// should correspond 1-1 with the successor inputs specified in
  /// `getSuccessorRegions`.
  ::mlir::OperandRange getSuccessorEntryOperands(::std::optional<unsigned> index);
  /// Returns the viable successors of a region at `index`, or the possible
  /// successors when branching from the parent op if `index` is None. These
  /// are the regions that may be selected during the flow of control. If
  /// `index` is None, `operands` is a set of optional attributes that
  /// either correspond to a constant value for each operand of this
  /// operation, or null if that operand is not a constant. If `index` is
  /// valid, `operands` corresponds to the entry values of the region at
  /// `index`. Only a region, i.e. a valid `index`, may use the parent
  /// operation as a successor. This method allows for describing which
  /// regions may be executed when entering an operation, and which regions
  /// are executed after having executed another region of the parent op. The
  /// successor region must be non-empty.
  void getSuccessorRegions(::std::optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions);
  /// Populates `invocationBounds` with the minimum and maximum number of
  /// times this operation will invoke the attached regions (assuming the
  /// regions yield normally, i.e. do not abort or invoke an infinite loop).
  /// The minimum number of invocations is at least 0. If the maximum number
  /// of invocations cannot be statically determined, then it will not have a
  /// value (i.e., it is set to `std::nullopt`).
  /// 
  /// `operands` is a set of optional attributes that either correspond to
  /// constant values for each operand of this operation or null if that
  /// operand is not a constant.
  /// 
  /// This method may be called speculatively on operations where the provided
  /// operands are not necessarily the same as the operation's current
  /// operands. This may occur in analyses that wish to determine "what would
  /// be the region invocations if these were the operands?"
  void getRegionInvocationBounds(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds);
  /// This method is called to compare types along control-flow edges. By
  /// default, the types are checked as equal.
  bool areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs);

    /// Convenience helper in case none of the operands is known.
    void getSuccessorRegions(std::optional<unsigned> index,
                             SmallVectorImpl<RegionSuccessor> &regions);

    /// Return `true` if control flow originating from the given region may
    /// eventually branch back to the same region. (Maybe after passing through
    /// other regions.)
    bool isRepetitiveRegion(unsigned index);
};
namespace detail {
  template <typename ConcreteOp>
  struct RegionBranchOpInterfaceTrait : public ::mlir::OpInterface<RegionBranchOpInterface, detail::RegionBranchOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns the operands of this operation used as the entry arguments when
    /// entering the region at `index`, which was specified as a successor of
    /// this operation by `getSuccessorRegions`, or the operands forwarded to
    /// the operation's results when it branches back to itself. These operands
    /// should correspond 1-1 with the successor inputs specified in
    /// `getSuccessorRegions`.
    ::mlir::OperandRange getSuccessorEntryOperands(::std::optional<unsigned> index) {
      auto operandEnd = this->getOperation()->operand_end();
        return ::mlir::OperandRange(operandEnd, operandEnd);
    }
    /// Populates `invocationBounds` with the minimum and maximum number of
    /// times this operation will invoke the attached regions (assuming the
    /// regions yield normally, i.e. do not abort or invoke an infinite loop).
    /// The minimum number of invocations is at least 0. If the maximum number
    /// of invocations cannot be statically determined, then it will not have a
    /// value (i.e., it is set to `std::nullopt`).
    /// 
    /// `operands` is a set of optional attributes that either correspond to
    /// constant values for each operand of this operation or null if that
    /// operand is not a constant.
    /// 
    /// This method may be called speculatively on operations where the provided
    /// operands are not necessarily the same as the operation's current
    /// operands. This may occur in analyses that wish to determine "what would
    /// be the region invocations if these were the operands?"
    void getRegionInvocationBounds(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds) {
      invocationBounds.append((*static_cast<ConcreteOp *>(this))->getNumRegions(),
                                  ::mlir::InvocationBounds::getUnknown());
    }
    /// This method is called to compare types along control-flow edges. By
    /// default, the types are checked as equal.
    bool areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs) {
      return lhs == rhs;
    }
    static ::mlir::LogicalResult verifyRegionTrait(::mlir::Operation *op) {
      static_assert(!ConcreteOp::template hasTrait<OpTrait::ZeroRegions>(),
                  "expected operation to have non-zero regions");
    return detail::verifyTypesAlongControlFlowEdges(op);
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class RegionBranchTerminatorOpInterface;
namespace detail {
struct RegionBranchTerminatorOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::MutableOperandRange (*getMutableSuccessorOperands)(const Concept *impl, ::mlir::Operation *, ::std::optional<unsigned>);
    ::mlir::OperandRange (*getSuccessorOperands)(const Concept *impl, ::mlir::Operation *, ::std::optional<unsigned>);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::RegionBranchTerminatorOpInterface;
    Model() : Concept{getMutableSuccessorOperands, getSuccessorOperands} {}

    static inline ::mlir::MutableOperandRange getMutableSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::std::optional<unsigned> index);
    static inline ::mlir::OperandRange getSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::std::optional<unsigned> index);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::RegionBranchTerminatorOpInterface;
    FallbackModel() : Concept{getMutableSuccessorOperands, getSuccessorOperands} {}

    static inline ::mlir::MutableOperandRange getMutableSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::std::optional<unsigned> index);
    static inline ::mlir::OperandRange getSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::std::optional<unsigned> index);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::OperandRange getSuccessorOperands(::mlir::Operation *tablegen_opaque_val, ::std::optional<unsigned> index) const;
  };
};template <typename ConcreteOp>
struct RegionBranchTerminatorOpInterfaceTrait;

} // namespace detail
class RegionBranchTerminatorOpInterface : public ::mlir::OpInterface<RegionBranchTerminatorOpInterface, detail::RegionBranchTerminatorOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<RegionBranchTerminatorOpInterface, detail::RegionBranchTerminatorOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::RegionBranchTerminatorOpInterfaceTrait<ConcreteOp> {};
  /// Returns a mutable range of operands that are semantically "returned" by
  /// passing them to the region successor given by `index`.  If `index` is
  /// None, this function returns the operands that are passed as a result to
  /// the parent operation.
  ::mlir::MutableOperandRange getMutableSuccessorOperands(::std::optional<unsigned> index);
  /// Returns a range of operands that are semantically "returned" by passing
  /// them to the region successor given by `index`.  If `index` is None, this
  /// function returns the operands that are passed as a result to the parent
  /// operation.
  ::mlir::OperandRange getSuccessorOperands(::std::optional<unsigned> index);
};
namespace detail {
  template <typename ConcreteOp>
  struct RegionBranchTerminatorOpInterfaceTrait : public ::mlir::OpInterface<RegionBranchTerminatorOpInterface, detail::RegionBranchTerminatorOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns a range of operands that are semantically "returned" by passing
    /// them to the region successor given by `index`.  If `index` is None, this
    /// function returns the operands that are passed as a result to the parent
    /// operation.
    ::mlir::OperandRange getSuccessorOperands(::std::optional<unsigned> index) {
      return (*static_cast<ConcreteOp *>(this)).getMutableSuccessorOperands(index);
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      static_assert(ConcreteOp::template hasTrait<OpTrait::IsTerminator>(),
                  "expected operation to be a terminator");
    static_assert(ConcreteOp::template hasTrait<OpTrait::ZeroResults>(),
                  "expected operation to have zero results");
    static_assert(ConcreteOp::template hasTrait<OpTrait::ZeroSuccessors>(),
                  "expected operation to have zero successors");
    return success();
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::mlir::SuccessorOperands detail::BranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorOperands(index);
}
template<typename ConcreteOp>
::std::optional<::mlir::BlockArgument> detail::BranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned operandIndex) {
  ::mlir::Operation *opaqueOp = (llvm::cast<ConcreteOp>(tablegen_opaque_val));
        for (unsigned i = 0, e = opaqueOp->getNumSuccessors(); i != e; ++i) {
          if (::std::optional<::mlir::BlockArgument> arg = ::mlir::detail::getBranchSuccessorArgument(
                (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorOperands(i), operandIndex,
                opaqueOp->getSuccessor(i)))
            return arg;
        }
        return ::std::nullopt;
}
template<typename ConcreteOp>
::mlir::Block *detail::BranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorForOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorForOperands(operands);
}
template<typename ConcreteOp>
bool detail::BranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).areTypesCompatible(lhs, rhs);
}
template<typename ConcreteOp>
::mlir::SuccessorOperands detail::BranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index) {
  return static_cast<const ConcreteOp *>(impl)->getSuccessorOperands(tablegen_opaque_val, index);
}
template<typename ConcreteOp>
::std::optional<::mlir::BlockArgument> detail::BranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSuccessorBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned operandIndex) {
  return static_cast<const ConcreteOp *>(impl)->getSuccessorBlockArgument(tablegen_opaque_val, operandIndex);
}
template<typename ConcreteOp>
::mlir::Block *detail::BranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSuccessorForOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands) {
  return static_cast<const ConcreteOp *>(impl)->getSuccessorForOperands(tablegen_opaque_val, operands);
}
template<typename ConcreteOp>
bool detail::BranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) {
  return static_cast<const ConcreteOp *>(impl)->areTypesCompatible(tablegen_opaque_val, lhs, rhs);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Block *detail::BranchOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getSuccessorForOperands(::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands) const {
return nullptr;
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::BranchOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::areTypesCompatible(::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) const {
return lhs == rhs;
}
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::mlir::OperandRange detail::RegionBranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorEntryOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::std::optional<unsigned> index) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorEntryOperands(index);
}
template<typename ConcreteOp>
void detail::RegionBranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::std::optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorRegions(index, operands, regions);
}
template<typename ConcreteOp>
void detail::RegionBranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getRegionInvocationBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRegionInvocationBounds(operands, invocationBounds);
}
template<typename ConcreteOp>
bool detail::RegionBranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).areTypesCompatible(lhs, rhs);
}
template<typename ConcreteOp>
::mlir::OperandRange detail::RegionBranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSuccessorEntryOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::std::optional<unsigned> index) {
  return static_cast<const ConcreteOp *>(impl)->getSuccessorEntryOperands(tablegen_opaque_val, index);
}
template<typename ConcreteOp>
void detail::RegionBranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSuccessorRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::std::optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions) {
  return static_cast<const ConcreteOp *>(impl)->getSuccessorRegions(tablegen_opaque_val, index, operands, regions);
}
template<typename ConcreteOp>
void detail::RegionBranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getRegionInvocationBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds) {
  return static_cast<const ConcreteOp *>(impl)->getRegionInvocationBounds(tablegen_opaque_val, operands, invocationBounds);
}
template<typename ConcreteOp>
bool detail::RegionBranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) {
  return static_cast<const ConcreteOp *>(impl)->areTypesCompatible(tablegen_opaque_val, lhs, rhs);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::OperandRange detail::RegionBranchOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getSuccessorEntryOperands(::mlir::Operation *tablegen_opaque_val, ::std::optional<unsigned> index) const {
auto operandEnd = this->getOperation()->operand_end();
        return ::mlir::OperandRange(operandEnd, operandEnd);
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::RegionBranchOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getRegionInvocationBounds(::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> &invocationBounds) const {
invocationBounds.append((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getNumRegions(),
                                  ::mlir::InvocationBounds::getUnknown());
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::RegionBranchOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::areTypesCompatible(::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) const {
return lhs == rhs;
}
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::mlir::MutableOperandRange detail::RegionBranchTerminatorOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMutableSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::std::optional<unsigned> index) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMutableSuccessorOperands(index);
}
template<typename ConcreteOp>
::mlir::OperandRange detail::RegionBranchTerminatorOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::std::optional<unsigned> index) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorOperands(index);
}
template<typename ConcreteOp>
::mlir::MutableOperandRange detail::RegionBranchTerminatorOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMutableSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::std::optional<unsigned> index) {
  return static_cast<const ConcreteOp *>(impl)->getMutableSuccessorOperands(tablegen_opaque_val, index);
}
template<typename ConcreteOp>
::mlir::OperandRange detail::RegionBranchTerminatorOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::std::optional<unsigned> index) {
  return static_cast<const ConcreteOp *>(impl)->getSuccessorOperands(tablegen_opaque_val, index);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::OperandRange detail::RegionBranchTerminatorOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getSuccessorOperands(::mlir::Operation *tablegen_opaque_val, ::std::optional<unsigned> index) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMutableSuccessorOperands(index);
}
} // namespace mlir
