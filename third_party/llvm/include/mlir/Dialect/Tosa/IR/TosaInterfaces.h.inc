/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

class TosaOp;
namespace detail {
struct TosaOpInterfaceTraits {
  struct Concept {
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = TosaOp;
    Model() : Concept{} {}

  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = TosaOp;
    FallbackModel() : Concept{} {}

  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};template <typename ConcreteOp>
struct TosaOpTrait;

} // namespace detail
class TosaOp : public ::mlir::OpInterface<TosaOp, detail::TosaOpInterfaceTraits> {
public:
  using ::mlir::OpInterface<TosaOp, detail::TosaOpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::TosaOpTrait<ConcreteOp> {};
};
namespace detail {
  template <typename ConcreteOp>
  struct TosaOpTrait : public ::mlir::OpInterface<TosaOp, detail::TosaOpInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
