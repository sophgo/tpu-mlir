/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace omp {
class ClauseDependAttr;
class ClauseMemoryOrderKindAttr;
class ClauseOrderKindAttr;
class ClauseProcBindKindAttr;
class ClauseScheduleKindAttr;
class ScheduleModifierAttr;
namespace detail {
struct ClauseDependAttrStorage;
} // namespace detail
class ClauseDependAttr : public ::mlir::Attribute::AttrBase<ClauseDependAttr, ::mlir::Attribute, detail::ClauseDependAttrStorage> {
public:
  using Base::Base;
public:
  static ClauseDependAttr get(::mlir::MLIRContext *context, ::mlir::omp::ClauseDepend value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"clause_depend"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::omp::ClauseDepend getValue() const;
};
namespace detail {
struct ClauseMemoryOrderKindAttrStorage;
} // namespace detail
class ClauseMemoryOrderKindAttr : public ::mlir::Attribute::AttrBase<ClauseMemoryOrderKindAttr, ::mlir::Attribute, detail::ClauseMemoryOrderKindAttrStorage> {
public:
  using Base::Base;
public:
  static ClauseMemoryOrderKindAttr get(::mlir::MLIRContext *context, ::mlir::omp::ClauseMemoryOrderKind value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"memoryorderkind"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::omp::ClauseMemoryOrderKind getValue() const;
};
namespace detail {
struct ClauseOrderKindAttrStorage;
} // namespace detail
class ClauseOrderKindAttr : public ::mlir::Attribute::AttrBase<ClauseOrderKindAttr, ::mlir::Attribute, detail::ClauseOrderKindAttrStorage> {
public:
  using Base::Base;
public:
  static ClauseOrderKindAttr get(::mlir::MLIRContext *context, ::mlir::omp::ClauseOrderKind value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"orderkind"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::omp::ClauseOrderKind getValue() const;
};
namespace detail {
struct ClauseProcBindKindAttrStorage;
} // namespace detail
class ClauseProcBindKindAttr : public ::mlir::Attribute::AttrBase<ClauseProcBindKindAttr, ::mlir::Attribute, detail::ClauseProcBindKindAttrStorage> {
public:
  using Base::Base;
public:
  static ClauseProcBindKindAttr get(::mlir::MLIRContext *context, ::mlir::omp::ClauseProcBindKind value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"procbindkind"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::omp::ClauseProcBindKind getValue() const;
};
namespace detail {
struct ClauseScheduleKindAttrStorage;
} // namespace detail
class ClauseScheduleKindAttr : public ::mlir::Attribute::AttrBase<ClauseScheduleKindAttr, ::mlir::Attribute, detail::ClauseScheduleKindAttrStorage> {
public:
  using Base::Base;
public:
  static ClauseScheduleKindAttr get(::mlir::MLIRContext *context, ::mlir::omp::ClauseScheduleKind value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"schedulekind"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::omp::ClauseScheduleKind getValue() const;
};
namespace detail {
struct ScheduleModifierAttrStorage;
} // namespace detail
class ScheduleModifierAttr : public ::mlir::Attribute::AttrBase<ScheduleModifierAttr, ::mlir::Attribute, detail::ScheduleModifierAttrStorage> {
public:
  using Base::Base;
public:
  static ScheduleModifierAttr get(::mlir::MLIRContext *context, ::mlir::omp::ScheduleModifier value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"sched_mod"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::omp::ScheduleModifier getValue() const;
};
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseDependAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseMemoryOrderKindAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseOrderKindAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseProcBindKindAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseScheduleKindAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ScheduleModifierAttr)

#endif  // GET_ATTRDEF_CLASSES

