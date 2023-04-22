/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class FunctionOpInterface;
namespace detail {
struct FunctionOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Type (*getFunctionType)(const Concept *impl, ::mlir::Operation *);
    void (*setFunctionTypeAttr)(const Concept *impl, ::mlir::Operation *, ::mlir::TypeAttr);
    ::mlir::ArrayAttr (*getArgAttrsAttr)(const Concept *impl, ::mlir::Operation *);
    ::mlir::ArrayAttr (*getResAttrsAttr)(const Concept *impl, ::mlir::Operation *);
    void (*setArgAttrsAttr)(const Concept *impl, ::mlir::Operation *, ::mlir::ArrayAttr);
    void (*setResAttrsAttr)(const Concept *impl, ::mlir::Operation *, ::mlir::ArrayAttr);
    ::mlir::Attribute (*removeArgAttrsAttr)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Attribute (*removeResAttrsAttr)(const Concept *impl, ::mlir::Operation *);
    ::llvm::ArrayRef<::mlir::Type> (*getArgumentTypes)(const Concept *impl, ::mlir::Operation *);
    ::llvm::ArrayRef<::mlir::Type> (*getResultTypes)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Type (*cloneTypeWith)(const Concept *impl, ::mlir::Operation *, ::mlir::TypeRange, ::mlir::TypeRange);
    ::mlir::LogicalResult (*verifyBody)(const Concept *impl, ::mlir::Operation *);
    ::mlir::LogicalResult (*verifyType)(const Concept *impl, ::mlir::Operation *);
    /// The base classes of this interface.
    const ::mlir::SymbolOpInterface::Concept *implSymbolOpInterface = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implSymbolOpInterface = interfaceMap.lookup<::mlir::SymbolOpInterface>();
      assert(implSymbolOpInterface && "`::mlir::FunctionOpInterface` expected its base interface `::mlir::SymbolOpInterface` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::FunctionOpInterface;
    Model() : Concept{getFunctionType, setFunctionTypeAttr, getArgAttrsAttr, getResAttrsAttr, setArgAttrsAttr, setResAttrsAttr, removeArgAttrsAttr, removeResAttrsAttr, getArgumentTypes, getResultTypes, cloneTypeWith, verifyBody, verifyType} {}

    static inline ::mlir::Type getFunctionType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setFunctionTypeAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeAttr type);
    static inline ::mlir::ArrayAttr getArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ArrayAttr getResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attrs);
    static inline void setResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attrs);
    static inline ::mlir::Attribute removeArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Attribute removeResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::ArrayRef<::mlir::Type> getArgumentTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::ArrayRef<::mlir::Type> getResultTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Type cloneTypeWith(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeRange inputs, ::mlir::TypeRange results);
    static inline ::mlir::LogicalResult verifyBody(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::LogicalResult verifyType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::FunctionOpInterface;
    FallbackModel() : Concept{getFunctionType, setFunctionTypeAttr, getArgAttrsAttr, getResAttrsAttr, setArgAttrsAttr, setResAttrsAttr, removeArgAttrsAttr, removeResAttrsAttr, getArgumentTypes, getResultTypes, cloneTypeWith, verifyBody, verifyType} {}

    static inline ::mlir::Type getFunctionType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setFunctionTypeAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeAttr type);
    static inline ::mlir::ArrayAttr getArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ArrayAttr getResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attrs);
    static inline void setResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attrs);
    static inline ::mlir::Attribute removeArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Attribute removeResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::ArrayRef<::mlir::Type> getArgumentTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::ArrayRef<::mlir::Type> getResultTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Type cloneTypeWith(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeRange inputs, ::mlir::TypeRange results);
    static inline ::mlir::LogicalResult verifyBody(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::LogicalResult verifyType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::Type cloneTypeWith(::mlir::Operation *tablegen_opaque_val, ::mlir::TypeRange inputs, ::mlir::TypeRange results) const;
    ::mlir::LogicalResult verifyBody(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::LogicalResult verifyType(::mlir::Operation *tablegen_opaque_val) const;
  };
};template <typename ConcreteOp>
struct FunctionOpInterfaceTrait;

} // namespace detail
class FunctionOpInterface : public ::mlir::OpInterface<FunctionOpInterface, detail::FunctionOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<FunctionOpInterface, detail::FunctionOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::FunctionOpInterfaceTrait<ConcreteOp> {};
  /// Returns the type of the function.
  ::mlir::Type getFunctionType();
  /// Set the type of the function. This method should perform an unsafe
  /// modification to the function type; it should not update argument or
  /// result attributes.
  void setFunctionTypeAttr(::mlir::TypeAttr type);
  /// Get the array of argument attribute dictionaries. The method should return
  /// an array attribute containing only dictionary attributes equal in number
  /// to the number of function arguments. Alternatively, the method can return
  /// null to indicate that the function has no argument attributes.
  ::mlir::ArrayAttr getArgAttrsAttr();
  /// Get the array of result attribute dictionaries. The method should return
  /// an array attribute containing only dictionary attributes equal in number
  /// to the number of function results. Alternatively, the method can return
  /// null to indicate that the function has no result attributes.
  ::mlir::ArrayAttr getResAttrsAttr();
  /// Set the array of argument attribute dictionaries.
  void setArgAttrsAttr(::mlir::ArrayAttr attrs);
  /// Set the array of result attribute dictionaries.
  void setResAttrsAttr(::mlir::ArrayAttr attrs);
  /// Remove the array of argument attribute dictionaries. This is the same as
  /// setting all argument attributes to an empty dictionary. The method should
  /// return the removed attribute.
  ::mlir::Attribute removeArgAttrsAttr();
  /// Remove the array of result attribute dictionaries. This is the same as
  /// setting all result attributes to an empty dictionary. The method should
  /// return the removed attribute.
  ::mlir::Attribute removeResAttrsAttr();
  /// Returns the function argument types based exclusively on
  /// the type (to allow for this method may be called on function
  /// declarations).
  ::llvm::ArrayRef<::mlir::Type> getArgumentTypes();
  /// Returns the function result types based exclusively on
  /// the type (to allow for this method may be called on function
  /// declarations).
  ::llvm::ArrayRef<::mlir::Type> getResultTypes();
  /// Returns a clone of the function type with the given argument and
  /// result types.
  /// 
  /// Note: The default implementation assumes the function type has
  ///       an appropriate clone method:
  ///         `Type clone(ArrayRef<Type> inputs, ArrayRef<Type> results)`
  ::mlir::Type cloneTypeWith(::mlir::TypeRange inputs, ::mlir::TypeRange results);
  /// Verify the contents of the body of this function.
  /// 
  /// Note: The default implementation merely checks that if the entry block
  /// exists, it has the same number and type of arguments as the function type.
  ::mlir::LogicalResult verifyBody();
  /// Verify the type attribute of the function for derived op-specific
  /// invariants.
  ::mlir::LogicalResult verifyType();

    /// Block list iterator types.
    using BlockListType = Region::BlockListType;
    using iterator = BlockListType::iterator;
    using reverse_iterator = BlockListType::reverse_iterator;

    /// Block argument iterator types.
    using BlockArgListType = Region::BlockArgListType;
    using args_iterator = BlockArgListType::iterator;

    //===------------------------------------------------------------------===//
    // Body Handling
    //===------------------------------------------------------------------===//

    /// Returns true if this function is external, i.e. it has no body.
    bool isExternal() { return empty(); }

    /// Return the region containing the body of this function.
    Region &getFunctionBody() { return (*this)->getRegion(0); }

    /// Delete all blocks from this function.
    void eraseBody() {
      getFunctionBody().dropAllReferences();
      getFunctionBody().getBlocks().clear();
    }

    /// Return the list of blocks within the function body.
    BlockListType &getBlocks() { return getFunctionBody().getBlocks(); }

    iterator begin() { return getFunctionBody().begin(); }
    iterator end() { return getFunctionBody().end(); }
    reverse_iterator rbegin() { return getFunctionBody().rbegin(); }
    reverse_iterator rend() { return getFunctionBody().rend(); }

    /// Returns true if this function has no blocks within the body.
    bool empty() { return getFunctionBody().empty(); }

    /// Push a new block to the back of the body region.
    void push_back(Block *block) { getFunctionBody().push_back(block); }

    /// Push a new block to the front of the body region.
    void push_front(Block *block) { getFunctionBody().push_front(block); }

    /// Return the last block in the body region.
    Block &back() { return getFunctionBody().back(); }

    /// Return the first block in the body region.
    Block &front() { return getFunctionBody().front(); }

    /// Add an entry block to an empty function, and set up the block arguments
    /// to match the signature of the function. The newly inserted entry block
    /// is returned.
    Block *addEntryBlock() {
      assert(empty() && "function already has an entry block");
      Block *entry = new Block();
      push_back(entry);

      // FIXME: Allow for passing in locations for these arguments instead of using
      // the operations location.
      ArrayRef<Type> inputTypes = (*this).getArgumentTypes();
      SmallVector<Location> locations(inputTypes.size(),
                                      (*this).getOperation()->getLoc());
      entry->addArguments(inputTypes, locations);
      return entry;
    }

    /// Add a normal block to the end of the function's block list. The function
    /// should at least already have an entry block.
    Block *addBlock() {
      assert(!empty() && "function should at least have an entry block");
      push_back(new Block());
      return &back();
    }

    //===------------------------------------------------------------------===//
    // Type Attribute Handling
    //===------------------------------------------------------------------===//

    /// Change the type of this function in place. This is an extremely dangerous
    /// operation and it is up to the caller to ensure that this is legal for
    /// this function, and to restore invariants:
    ///  - the entry block args must be updated to match the function params.
    ///  - the argument/result attributes may need an update: if the new type
    ///    has less parameters we drop the extra attributes, if there are more
    ///    parameters they won't have any attributes.
    void setType(Type newType) {
      function_interface_impl::setFunctionType(this->getOperation(), newType);
    }

    //===------------------------------------------------------------------===//
    // Argument and Result Handling
    //===------------------------------------------------------------------===//

    /// Returns the number of function arguments.
    unsigned getNumArguments() { return (*this).getArgumentTypes().size(); }

    /// Returns the number of function results.
    unsigned getNumResults() { return (*this).getResultTypes().size(); }

    /// Returns the entry block function argument at the given index.
    BlockArgument getArgument(unsigned idx) {
      return getFunctionBody().getArgument(idx);
    }

    /// Support argument iteration.
    args_iterator args_begin() { return getFunctionBody().args_begin(); }
    args_iterator args_end() { return getFunctionBody().args_end(); }
    BlockArgListType getArguments() { return getFunctionBody().getArguments(); }

    /// Insert a single argument of type `argType` with attributes `argAttrs` and
    /// location `argLoc` at `argIndex`.
    void insertArgument(unsigned argIndex, Type argType, DictionaryAttr argAttrs,
                        Location argLoc) {
      insertArguments({argIndex}, {argType}, {argAttrs}, {argLoc});
    }

    /// Inserts arguments with the listed types, attributes, and locations at the
    /// listed indices. `argIndices` must be sorted. Arguments are inserted in the
    /// order they are listed, such that arguments with identical index will
    /// appear in the same order that they were listed here.
    void insertArguments(ArrayRef<unsigned> argIndices, TypeRange argTypes,
                        ArrayRef<DictionaryAttr> argAttrs,
                        ArrayRef<Location> argLocs) {
      unsigned originalNumArgs = (*this).getNumArguments();
      Type newType = (*this).getTypeWithArgsAndResults(
          argIndices, argTypes, /*resultIndices=*/{}, /*resultTypes=*/{});
      function_interface_impl::insertFunctionArguments(
          this->getOperation(), argIndices, argTypes, argAttrs, argLocs,
          originalNumArgs, newType);
    }

    /// Insert a single result of type `resultType` at `resultIndex`.
    void insertResult(unsigned resultIndex, Type resultType,
                      DictionaryAttr resultAttrs) {
      insertResults({resultIndex}, {resultType}, {resultAttrs});
    }

    /// Inserts results with the listed types at the listed indices.
    /// `resultIndices` must be sorted. Results are inserted in the order they are
    /// listed, such that results with identical index will appear in the same
    /// order that they were listed here.
    void insertResults(ArrayRef<unsigned> resultIndices, TypeRange resultTypes,
                      ArrayRef<DictionaryAttr> resultAttrs) {
      unsigned originalNumResults = (*this).getNumResults();
      Type newType = (*this).getTypeWithArgsAndResults(
        /*argIndices=*/{}, /*argTypes=*/{}, resultIndices, resultTypes);
      function_interface_impl::insertFunctionResults(
          this->getOperation(), resultIndices, resultTypes, resultAttrs,
          originalNumResults, newType);
    }

    /// Erase a single argument at `argIndex`.
    void eraseArgument(unsigned argIndex) {
      BitVector argsToErase((*this).getNumArguments());
      argsToErase.set(argIndex);
      eraseArguments(argsToErase);
    }

    /// Erases the arguments listed in `argIndices`.
    void eraseArguments(const BitVector &argIndices) {
      Type newType = (*this).getTypeWithoutArgs(argIndices);
      function_interface_impl::eraseFunctionArguments(
        this->getOperation(), argIndices, newType);
    }

    /// Erase a single result at `resultIndex`.
    void eraseResult(unsigned resultIndex) {
      BitVector resultsToErase((*this).getNumResults());
      resultsToErase.set(resultIndex);
      eraseResults(resultsToErase);
    }

    /// Erases the results listed in `resultIndices`.
    void eraseResults(const BitVector &resultIndices) {
      Type newType = (*this).getTypeWithoutResults(resultIndices);
      function_interface_impl::eraseFunctionResults(
          this->getOperation(), resultIndices, newType);
    }

    /// Return the type of this function with the specified arguments and
    /// results inserted. This is used to update the function's signature in
    /// the `insertArguments` and `insertResults` methods. The arrays must be
    /// sorted by increasing index.
    Type getTypeWithArgsAndResults(
      ArrayRef<unsigned> argIndices, TypeRange argTypes,
      ArrayRef<unsigned> resultIndices, TypeRange resultTypes) {
      SmallVector<Type> argStorage, resultStorage;
      TypeRange newArgTypes = function_interface_impl::insertTypesInto(
          (*this).getArgumentTypes(), argIndices, argTypes, argStorage);
      TypeRange newResultTypes = function_interface_impl::insertTypesInto(
          (*this).getResultTypes(), resultIndices, resultTypes, resultStorage);
      return (*this).cloneTypeWith(newArgTypes, newResultTypes);
    }

    /// Return the type of this function without the specified arguments and
    /// results. This is used to update the function's signature in the
    /// `eraseArguments` and `eraseResults` methods.
    Type getTypeWithoutArgsAndResults(
      const BitVector &argIndices, const BitVector &resultIndices) {
      SmallVector<Type> argStorage, resultStorage;
      TypeRange newArgTypes = function_interface_impl::filterTypesOut(
          (*this).getArgumentTypes(), argIndices, argStorage);
      TypeRange newResultTypes = function_interface_impl::filterTypesOut(
          (*this).getResultTypes(), resultIndices, resultStorage);
      return (*this).cloneTypeWith(newArgTypes, newResultTypes);
    }
    Type getTypeWithoutArgs(const BitVector &argIndices) {
      SmallVector<Type> argStorage;
      TypeRange newArgTypes = function_interface_impl::filterTypesOut(
          (*this).getArgumentTypes(), argIndices, argStorage);
      return (*this).cloneTypeWith(newArgTypes, (*this).getResultTypes());
    }
    Type getTypeWithoutResults(const BitVector &resultIndices) {
      SmallVector<Type> resultStorage;
      TypeRange newResultTypes = function_interface_impl::filterTypesOut(
          (*this).getResultTypes(), resultIndices, resultStorage);
      return (*this).cloneTypeWith((*this).getArgumentTypes(), newResultTypes);
    }

    //===------------------------------------------------------------------===//
    // Argument Attributes
    //===------------------------------------------------------------------===//

    /// Return all of the attributes for the argument at 'index'.
    ArrayRef<NamedAttribute> getArgAttrs(unsigned index) {
      return function_interface_impl::getArgAttrs(this->getOperation(), index);
    }

    /// Return an ArrayAttr containing all argument attribute dictionaries of
    /// this function, or nullptr if no arguments have attributes.
    ArrayAttr getAllArgAttrs() { return (*this).getArgAttrsAttr(); }

    /// Return all argument attributes of this function.
    void getAllArgAttrs(SmallVectorImpl<DictionaryAttr> &result) {
      if (ArrayAttr argAttrs = getAllArgAttrs()) {
        auto argAttrRange = argAttrs.template getAsRange<DictionaryAttr>();
        result.append(argAttrRange.begin(), argAttrRange.end());
      } else {
        result.append((*this).getNumArguments(),
                      DictionaryAttr::get(this->getOperation()->getContext()));
      }
    }

    /// Return the specified attribute, if present, for the argument at 'index',
    /// null otherwise.
    Attribute getArgAttr(unsigned index, StringAttr name) {
      auto argDict = getArgAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }
    Attribute getArgAttr(unsigned index, StringRef name) {
      auto argDict = getArgAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }

    template <typename AttrClass>
    AttrClass getArgAttrOfType(unsigned index, StringAttr name) {
      return getArgAttr(index, name).template dyn_cast_or_null<AttrClass>();
    }
    template <typename AttrClass>
    AttrClass getArgAttrOfType(unsigned index, StringRef name) {
      return getArgAttr(index, name).template dyn_cast_or_null<AttrClass>();
    }

    /// Set the attributes held by the argument at 'index'.
    void setArgAttrs(unsigned index, ArrayRef<NamedAttribute> attributes) {
      function_interface_impl::setArgAttrs((*this), index, attributes);
    }

    /// Set the attributes held by the argument at 'index'. `attributes` may be
    /// null, in which case any existing argument attributes are removed.
    void setArgAttrs(unsigned index, DictionaryAttr attributes) {
      function_interface_impl::setArgAttrs((*this), index, attributes);
    }
    void setAllArgAttrs(ArrayRef<DictionaryAttr> attributes) {
      assert(attributes.size() == (*this).getNumArguments());
      function_interface_impl::setAllArgAttrDicts(this->getOperation(), attributes);
    }
    void setAllArgAttrs(ArrayRef<Attribute> attributes) {
      assert(attributes.size() == (*this).getNumArguments());
      function_interface_impl::setAllArgAttrDicts(this->getOperation(), attributes);
    }
    void setAllArgAttrs(ArrayAttr attributes) {
      assert(attributes.size() == (*this).getNumArguments());
      (*this).setArgAttrsAttr(attributes);
    }

    /// If the an attribute exists with the specified name, change it to the new
    /// value. Otherwise, add a new attribute with the specified name/value.
    void setArgAttr(unsigned index, StringAttr name, Attribute value) {
      function_interface_impl::setArgAttr((*this), index, name, value);
    }
    void setArgAttr(unsigned index, StringRef name, Attribute value) {
      setArgAttr(index,
                 StringAttr::get(this->getOperation()->getContext(), name),
                 value);
    }

    /// Remove the attribute 'name' from the argument at 'index'. Return the
    /// attribute that was erased, or nullptr if there was no attribute with
    /// such name.
    Attribute removeArgAttr(unsigned index, StringAttr name) {
      return function_interface_impl::removeArgAttr((*this), index, name);
    }
    Attribute removeArgAttr(unsigned index, StringRef name) {
      return removeArgAttr(
          index, StringAttr::get(this->getOperation()->getContext(), name));
    }

    //===------------------------------------------------------------------===//
    // Result Attributes
    //===------------------------------------------------------------------===//

    /// Return all of the attributes for the result at 'index'.
    ArrayRef<NamedAttribute> getResultAttrs(unsigned index) {
      return function_interface_impl::getResultAttrs(this->getOperation(), index);
    }

    /// Return an ArrayAttr containing all result attribute dictionaries of this
    /// function, or nullptr if no result have attributes.
    ArrayAttr getAllResultAttrs() { return (*this).getResAttrsAttr(); }

    /// Return all result attributes of this function.
    void getAllResultAttrs(SmallVectorImpl<DictionaryAttr> &result) {
      if (ArrayAttr argAttrs = getAllResultAttrs()) {
        auto argAttrRange = argAttrs.template getAsRange<DictionaryAttr>();
        result.append(argAttrRange.begin(), argAttrRange.end());
      } else {
        result.append((*this).getNumResults(),
                      DictionaryAttr::get(this->getOperation()->getContext()));
      }
    }

    /// Return the specified attribute, if present, for the result at 'index',
    /// null otherwise.
    Attribute getResultAttr(unsigned index, StringAttr name) {
      auto argDict = getResultAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }
    Attribute getResultAttr(unsigned index, StringRef name) {
      auto argDict = getResultAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }

    template <typename AttrClass>
    AttrClass getResultAttrOfType(unsigned index, StringAttr name) {
      return getResultAttr(index, name).template dyn_cast_or_null<AttrClass>();
    }
    template <typename AttrClass>
    AttrClass getResultAttrOfType(unsigned index, StringRef name) {
      return getResultAttr(index, name).template dyn_cast_or_null<AttrClass>();
    }

    /// Set the attributes held by the result at 'index'.
    void setResultAttrs(unsigned index, ArrayRef<NamedAttribute> attributes) {
      function_interface_impl::setResultAttrs((*this), index, attributes);
    }

    /// Set the attributes held by the result at 'index'. `attributes` may be
    /// null, in which case any existing argument attributes are removed.
    void setResultAttrs(unsigned index, DictionaryAttr attributes) {
      function_interface_impl::setResultAttrs((*this), index, attributes);
    }
    void setAllResultAttrs(ArrayRef<DictionaryAttr> attributes) {
      assert(attributes.size() == (*this).getNumResults());
      function_interface_impl::setAllResultAttrDicts(
        this->getOperation(), attributes);
    }
    void setAllResultAttrs(ArrayRef<Attribute> attributes) {
      assert(attributes.size() == (*this).getNumResults());
      function_interface_impl::setAllResultAttrDicts(
        this->getOperation(), attributes);
    }
    void setAllResultAttrs(ArrayAttr attributes) {
      assert(attributes.size() == (*this).getNumResults());
      (*this).setResAttrsAttr(attributes);
    }

    /// If the an attribute exists with the specified name, change it to the new
    /// value. Otherwise, add a new attribute with the specified name/value.
    void setResultAttr(unsigned index, StringAttr name, Attribute value) {
      function_interface_impl::setResultAttr((*this), index, name, value);
    }
    void setResultAttr(unsigned index, StringRef name, Attribute value) {
      setResultAttr(index,
                    StringAttr::get(this->getOperation()->getContext(), name),
                    value);
    }

    /// Remove the attribute 'name' from the result at 'index'. Return the
    /// attribute that was erased, or nullptr if there was no attribute with
    /// such name.
    Attribute removeResultAttr(unsigned index, StringAttr name) {
      return function_interface_impl::removeResultAttr((*this), index, name);
    }

    /// Returns the dictionary attribute corresponding to the argument at
    /// 'index'. If there are no argument attributes at 'index', a null
    /// attribute is returned.
    DictionaryAttr getArgAttrDict(unsigned index) {
      assert(index < (*this).getNumArguments() && "invalid argument number");
      return function_interface_impl::getArgAttrDict(this->getOperation(), index);
    }

    /// Returns the dictionary attribute corresponding to the result at 'index'.
    /// If there are no result attributes at 'index', a null attribute is
    /// returned.
    DictionaryAttr getResultAttrDict(unsigned index) {
      assert(index < (*this).getNumResults() && "invalid result number");
      return function_interface_impl::getResultAttrDict(this->getOperation(), index);
    }
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::SymbolOpInterface
  //===----------------------------------------------------------------===//

  operator ::mlir::SymbolOpInterface () const {
    return ::mlir::SymbolOpInterface(*this, getImpl()->implSymbolOpInterface);
  }

  /// Returns the name of this symbol.
  StringAttr getNameAttr();
  /// Sets the name of this symbol.
  void setName(::mlir::StringAttr name);
  /// Gets the visibility of this symbol.
  mlir::SymbolTable::Visibility getVisibility();
  /// Returns true if this symbol has nested visibility.
  bool isNested();
  /// Returns true if this symbol has private visibility.
  bool isPrivate();
  /// Returns true if this symbol has public visibility.
  bool isPublic();
  /// Sets the visibility of this symbol.
  void setVisibility(mlir::SymbolTable::Visibility vis);
  /// Sets the visibility of this symbol to be nested.
  void setNested();
  /// Sets the visibility of this symbol to be private.
  void setPrivate();
  /// Sets the visibility of this symbol to be public.
  void setPublic();
  /// Get all of the uses of the current symbol that are nested within the
  /// given operation 'from'.
  /// Note: See mlir::SymbolTable::getSymbolUses for more details.
  ::std::optional<::mlir::SymbolTable::UseRange> getSymbolUses(::mlir::Operation * from);
  /// Return if the current symbol is known to have no uses that are nested
  /// within the given operation 'from'.
  /// Note: See mlir::SymbolTable::symbolKnownUseEmpty for more details.
  bool symbolKnownUseEmpty(::mlir::Operation * from);
  /// Attempt to replace all uses of the current symbol with the provided
  /// symbol 'newSymbol' that are nested within the given operation 'from'.
  /// Note: See mlir::SymbolTable::replaceAllSymbolUses for more details.
  ::mlir::LogicalResult replaceAllSymbolUses(::mlir::StringAttr newSymbol, ::mlir::Operation * from);
  /// Returns true if this operation optionally defines a symbol based on the
  /// presence of the symbol name.
  bool isOptionalSymbol();
  /// Returns true if this operation can be discarded if it has no remaining
  /// symbol uses.
  bool canDiscardOnUseEmpty();
  /// Returns true if this operation is a declaration of a symbol (as opposed
  /// to a definition).
  bool isDeclaration();

    using Visibility = mlir::SymbolTable::Visibility;

    /// Convenience version of `getNameAttr` that returns a StringRef.
    StringRef getName() {
      return getNameAttr().getValue();
    }

    /// Convenience version of `setName` that take a StringRef.
    void setName(StringRef name) {
      setName(StringAttr::get((*this)->getContext(), name));
    }
};
namespace detail {
  template <typename ConcreteOp>
  struct FunctionOpInterfaceTrait : public ::mlir::OpInterface<FunctionOpInterface, detail::FunctionOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns a clone of the function type with the given argument and
    /// result types.
    /// 
    /// Note: The default implementation assumes the function type has
    ///       an appropriate clone method:
    ///         `Type clone(ArrayRef<Type> inputs, ArrayRef<Type> results)`
    ::mlir::Type cloneTypeWith(::mlir::TypeRange inputs, ::mlir::TypeRange results) {
      return (*static_cast<ConcreteOp *>(this)).getFunctionType().clone(inputs, results);
    }
    /// Verify the contents of the body of this function.
    /// 
    /// Note: The default implementation merely checks that if the entry block
    /// exists, it has the same number and type of arguments as the function type.
    ::mlir::LogicalResult verifyBody() {
      if ((*static_cast<ConcreteOp *>(this)).isExternal())
        return success();
      ArrayRef<Type> fnInputTypes = (*static_cast<ConcreteOp *>(this)).getArgumentTypes();
      // NOTE: This should just be (*static_cast<ConcreteOp *>(this)).front() but access generically
      // because the interface methods defined here may be shadowed in
      // arbitrary ways. https://github.com/llvm/llvm-project/issues/54807
      Block &entryBlock = (*static_cast<ConcreteOp *>(this))->getRegion(0).front();

      unsigned numArguments = fnInputTypes.size();
      if (entryBlock.getNumArguments() != numArguments)
        return (*static_cast<ConcreteOp *>(this)).emitOpError("entry block must have ")
              << numArguments << " arguments to match function signature";

      for (unsigned i = 0, e = fnInputTypes.size(); i != e; ++i) {
        Type argType = entryBlock.getArgument(i).getType();
        if (fnInputTypes[i] != argType) {
          return (*static_cast<ConcreteOp *>(this)).emitOpError("type of entry block argument #")
                << i << '(' << argType
                << ") must match the type of the corresponding argument in "
                << "function signature(" << fnInputTypes[i] << ')';
        }
      }

      return success();
    }
    /// Verify the type attribute of the function for derived op-specific
    /// invariants.
    ::mlir::LogicalResult verifyType() {
      return success();
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return function_interface_impl::verifyTrait(cast<ConcreteOp>(op));
    }

    //===------------------------------------------------------------------===//
    // Builders
    //===------------------------------------------------------------------===//

    /// Build the function with the given name, attributes, and type. This
    /// builder also inserts an entry block into the function body with the
    /// given argument types.
    static void buildWithEntryBlock(
        OpBuilder &builder, OperationState &state, StringRef name, Type type,
        ArrayRef<NamedAttribute> attrs, TypeRange inputTypes) {
      state.addAttribute(SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr(name));
      state.addAttribute(ConcreteOp::getFunctionTypeAttrName(state.name),
                        TypeAttr::get(type));
      state.attributes.append(attrs.begin(), attrs.end());

      // Add the function body.
      Region *bodyRegion = state.addRegion();
      Block *body = new Block();
      bodyRegion->push_back(body);
      for (Type input : inputTypes)
        body->addArgument(input, state.location);
    }
  

    /// Block list iterator types.
    using BlockListType = Region::BlockListType;
    using iterator = BlockListType::iterator;
    using reverse_iterator = BlockListType::reverse_iterator;

    /// Block argument iterator types.
    using BlockArgListType = Region::BlockArgListType;
    using args_iterator = BlockArgListType::iterator;

    //===------------------------------------------------------------------===//
    // Body Handling
    //===------------------------------------------------------------------===//

    /// Returns true if this function is external, i.e. it has no body.
    bool isExternal() { return empty(); }

    /// Return the region containing the body of this function.
    Region &getFunctionBody() { return (*static_cast<ConcreteOp *>(this))->getRegion(0); }

    /// Delete all blocks from this function.
    void eraseBody() {
      getFunctionBody().dropAllReferences();
      getFunctionBody().getBlocks().clear();
    }

    /// Return the list of blocks within the function body.
    BlockListType &getBlocks() { return getFunctionBody().getBlocks(); }

    iterator begin() { return getFunctionBody().begin(); }
    iterator end() { return getFunctionBody().end(); }
    reverse_iterator rbegin() { return getFunctionBody().rbegin(); }
    reverse_iterator rend() { return getFunctionBody().rend(); }

    /// Returns true if this function has no blocks within the body.
    bool empty() { return getFunctionBody().empty(); }

    /// Push a new block to the back of the body region.
    void push_back(Block *block) { getFunctionBody().push_back(block); }

    /// Push a new block to the front of the body region.
    void push_front(Block *block) { getFunctionBody().push_front(block); }

    /// Return the last block in the body region.
    Block &back() { return getFunctionBody().back(); }

    /// Return the first block in the body region.
    Block &front() { return getFunctionBody().front(); }

    /// Add an entry block to an empty function, and set up the block arguments
    /// to match the signature of the function. The newly inserted entry block
    /// is returned.
    Block *addEntryBlock() {
      assert(empty() && "function already has an entry block");
      Block *entry = new Block();
      push_back(entry);

      // FIXME: Allow for passing in locations for these arguments instead of using
      // the operations location.
      ArrayRef<Type> inputTypes = (*static_cast<ConcreteOp *>(this)).getArgumentTypes();
      SmallVector<Location> locations(inputTypes.size(),
                                      (*static_cast<ConcreteOp *>(this)).getOperation()->getLoc());
      entry->addArguments(inputTypes, locations);
      return entry;
    }

    /// Add a normal block to the end of the function's block list. The function
    /// should at least already have an entry block.
    Block *addBlock() {
      assert(!empty() && "function should at least have an entry block");
      push_back(new Block());
      return &back();
    }

    //===------------------------------------------------------------------===//
    // Type Attribute Handling
    //===------------------------------------------------------------------===//

    /// Change the type of this function in place. This is an extremely dangerous
    /// operation and it is up to the caller to ensure that this is legal for
    /// this function, and to restore invariants:
    ///  - the entry block args must be updated to match the function params.
    ///  - the argument/result attributes may need an update: if the new type
    ///    has less parameters we drop the extra attributes, if there are more
    ///    parameters they won't have any attributes.
    void setType(Type newType) {
      function_interface_impl::setFunctionType(this->getOperation(), newType);
    }

    //===------------------------------------------------------------------===//
    // Argument and Result Handling
    //===------------------------------------------------------------------===//

    /// Returns the number of function arguments.
    unsigned getNumArguments() { return (*static_cast<ConcreteOp *>(this)).getArgumentTypes().size(); }

    /// Returns the number of function results.
    unsigned getNumResults() { return (*static_cast<ConcreteOp *>(this)).getResultTypes().size(); }

    /// Returns the entry block function argument at the given index.
    BlockArgument getArgument(unsigned idx) {
      return getFunctionBody().getArgument(idx);
    }

    /// Support argument iteration.
    args_iterator args_begin() { return getFunctionBody().args_begin(); }
    args_iterator args_end() { return getFunctionBody().args_end(); }
    BlockArgListType getArguments() { return getFunctionBody().getArguments(); }

    /// Insert a single argument of type `argType` with attributes `argAttrs` and
    /// location `argLoc` at `argIndex`.
    void insertArgument(unsigned argIndex, Type argType, DictionaryAttr argAttrs,
                        Location argLoc) {
      insertArguments({argIndex}, {argType}, {argAttrs}, {argLoc});
    }

    /// Inserts arguments with the listed types, attributes, and locations at the
    /// listed indices. `argIndices` must be sorted. Arguments are inserted in the
    /// order they are listed, such that arguments with identical index will
    /// appear in the same order that they were listed here.
    void insertArguments(ArrayRef<unsigned> argIndices, TypeRange argTypes,
                        ArrayRef<DictionaryAttr> argAttrs,
                        ArrayRef<Location> argLocs) {
      unsigned originalNumArgs = (*static_cast<ConcreteOp *>(this)).getNumArguments();
      Type newType = (*static_cast<ConcreteOp *>(this)).getTypeWithArgsAndResults(
          argIndices, argTypes, /*resultIndices=*/{}, /*resultTypes=*/{});
      function_interface_impl::insertFunctionArguments(
          this->getOperation(), argIndices, argTypes, argAttrs, argLocs,
          originalNumArgs, newType);
    }

    /// Insert a single result of type `resultType` at `resultIndex`.
    void insertResult(unsigned resultIndex, Type resultType,
                      DictionaryAttr resultAttrs) {
      insertResults({resultIndex}, {resultType}, {resultAttrs});
    }

    /// Inserts results with the listed types at the listed indices.
    /// `resultIndices` must be sorted. Results are inserted in the order they are
    /// listed, such that results with identical index will appear in the same
    /// order that they were listed here.
    void insertResults(ArrayRef<unsigned> resultIndices, TypeRange resultTypes,
                      ArrayRef<DictionaryAttr> resultAttrs) {
      unsigned originalNumResults = (*static_cast<ConcreteOp *>(this)).getNumResults();
      Type newType = (*static_cast<ConcreteOp *>(this)).getTypeWithArgsAndResults(
        /*argIndices=*/{}, /*argTypes=*/{}, resultIndices, resultTypes);
      function_interface_impl::insertFunctionResults(
          this->getOperation(), resultIndices, resultTypes, resultAttrs,
          originalNumResults, newType);
    }

    /// Erase a single argument at `argIndex`.
    void eraseArgument(unsigned argIndex) {
      BitVector argsToErase((*static_cast<ConcreteOp *>(this)).getNumArguments());
      argsToErase.set(argIndex);
      eraseArguments(argsToErase);
    }

    /// Erases the arguments listed in `argIndices`.
    void eraseArguments(const BitVector &argIndices) {
      Type newType = (*static_cast<ConcreteOp *>(this)).getTypeWithoutArgs(argIndices);
      function_interface_impl::eraseFunctionArguments(
        this->getOperation(), argIndices, newType);
    }

    /// Erase a single result at `resultIndex`.
    void eraseResult(unsigned resultIndex) {
      BitVector resultsToErase((*static_cast<ConcreteOp *>(this)).getNumResults());
      resultsToErase.set(resultIndex);
      eraseResults(resultsToErase);
    }

    /// Erases the results listed in `resultIndices`.
    void eraseResults(const BitVector &resultIndices) {
      Type newType = (*static_cast<ConcreteOp *>(this)).getTypeWithoutResults(resultIndices);
      function_interface_impl::eraseFunctionResults(
          this->getOperation(), resultIndices, newType);
    }

    /// Return the type of this function with the specified arguments and
    /// results inserted. This is used to update the function's signature in
    /// the `insertArguments` and `insertResults` methods. The arrays must be
    /// sorted by increasing index.
    Type getTypeWithArgsAndResults(
      ArrayRef<unsigned> argIndices, TypeRange argTypes,
      ArrayRef<unsigned> resultIndices, TypeRange resultTypes) {
      SmallVector<Type> argStorage, resultStorage;
      TypeRange newArgTypes = function_interface_impl::insertTypesInto(
          (*static_cast<ConcreteOp *>(this)).getArgumentTypes(), argIndices, argTypes, argStorage);
      TypeRange newResultTypes = function_interface_impl::insertTypesInto(
          (*static_cast<ConcreteOp *>(this)).getResultTypes(), resultIndices, resultTypes, resultStorage);
      return (*static_cast<ConcreteOp *>(this)).cloneTypeWith(newArgTypes, newResultTypes);
    }

    /// Return the type of this function without the specified arguments and
    /// results. This is used to update the function's signature in the
    /// `eraseArguments` and `eraseResults` methods.
    Type getTypeWithoutArgsAndResults(
      const BitVector &argIndices, const BitVector &resultIndices) {
      SmallVector<Type> argStorage, resultStorage;
      TypeRange newArgTypes = function_interface_impl::filterTypesOut(
          (*static_cast<ConcreteOp *>(this)).getArgumentTypes(), argIndices, argStorage);
      TypeRange newResultTypes = function_interface_impl::filterTypesOut(
          (*static_cast<ConcreteOp *>(this)).getResultTypes(), resultIndices, resultStorage);
      return (*static_cast<ConcreteOp *>(this)).cloneTypeWith(newArgTypes, newResultTypes);
    }
    Type getTypeWithoutArgs(const BitVector &argIndices) {
      SmallVector<Type> argStorage;
      TypeRange newArgTypes = function_interface_impl::filterTypesOut(
          (*static_cast<ConcreteOp *>(this)).getArgumentTypes(), argIndices, argStorage);
      return (*static_cast<ConcreteOp *>(this)).cloneTypeWith(newArgTypes, (*static_cast<ConcreteOp *>(this)).getResultTypes());
    }
    Type getTypeWithoutResults(const BitVector &resultIndices) {
      SmallVector<Type> resultStorage;
      TypeRange newResultTypes = function_interface_impl::filterTypesOut(
          (*static_cast<ConcreteOp *>(this)).getResultTypes(), resultIndices, resultStorage);
      return (*static_cast<ConcreteOp *>(this)).cloneTypeWith((*static_cast<ConcreteOp *>(this)).getArgumentTypes(), newResultTypes);
    }

    //===------------------------------------------------------------------===//
    // Argument Attributes
    //===------------------------------------------------------------------===//

    /// Return all of the attributes for the argument at 'index'.
    ArrayRef<NamedAttribute> getArgAttrs(unsigned index) {
      return function_interface_impl::getArgAttrs(this->getOperation(), index);
    }

    /// Return an ArrayAttr containing all argument attribute dictionaries of
    /// this function, or nullptr if no arguments have attributes.
    ArrayAttr getAllArgAttrs() { return (*static_cast<ConcreteOp *>(this)).getArgAttrsAttr(); }

    /// Return all argument attributes of this function.
    void getAllArgAttrs(SmallVectorImpl<DictionaryAttr> &result) {
      if (ArrayAttr argAttrs = getAllArgAttrs()) {
        auto argAttrRange = argAttrs.template getAsRange<DictionaryAttr>();
        result.append(argAttrRange.begin(), argAttrRange.end());
      } else {
        result.append((*static_cast<ConcreteOp *>(this)).getNumArguments(),
                      DictionaryAttr::get(this->getOperation()->getContext()));
      }
    }

    /// Return the specified attribute, if present, for the argument at 'index',
    /// null otherwise.
    Attribute getArgAttr(unsigned index, StringAttr name) {
      auto argDict = getArgAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }
    Attribute getArgAttr(unsigned index, StringRef name) {
      auto argDict = getArgAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }

    template <typename AttrClass>
    AttrClass getArgAttrOfType(unsigned index, StringAttr name) {
      return getArgAttr(index, name).template dyn_cast_or_null<AttrClass>();
    }
    template <typename AttrClass>
    AttrClass getArgAttrOfType(unsigned index, StringRef name) {
      return getArgAttr(index, name).template dyn_cast_or_null<AttrClass>();
    }

    /// Set the attributes held by the argument at 'index'.
    void setArgAttrs(unsigned index, ArrayRef<NamedAttribute> attributes) {
      function_interface_impl::setArgAttrs((*static_cast<ConcreteOp *>(this)), index, attributes);
    }

    /// Set the attributes held by the argument at 'index'. `attributes` may be
    /// null, in which case any existing argument attributes are removed.
    void setArgAttrs(unsigned index, DictionaryAttr attributes) {
      function_interface_impl::setArgAttrs((*static_cast<ConcreteOp *>(this)), index, attributes);
    }
    void setAllArgAttrs(ArrayRef<DictionaryAttr> attributes) {
      assert(attributes.size() == (*static_cast<ConcreteOp *>(this)).getNumArguments());
      function_interface_impl::setAllArgAttrDicts(this->getOperation(), attributes);
    }
    void setAllArgAttrs(ArrayRef<Attribute> attributes) {
      assert(attributes.size() == (*static_cast<ConcreteOp *>(this)).getNumArguments());
      function_interface_impl::setAllArgAttrDicts(this->getOperation(), attributes);
    }
    void setAllArgAttrs(ArrayAttr attributes) {
      assert(attributes.size() == (*static_cast<ConcreteOp *>(this)).getNumArguments());
      (*static_cast<ConcreteOp *>(this)).setArgAttrsAttr(attributes);
    }

    /// If the an attribute exists with the specified name, change it to the new
    /// value. Otherwise, add a new attribute with the specified name/value.
    void setArgAttr(unsigned index, StringAttr name, Attribute value) {
      function_interface_impl::setArgAttr((*static_cast<ConcreteOp *>(this)), index, name, value);
    }
    void setArgAttr(unsigned index, StringRef name, Attribute value) {
      setArgAttr(index,
                 StringAttr::get(this->getOperation()->getContext(), name),
                 value);
    }

    /// Remove the attribute 'name' from the argument at 'index'. Return the
    /// attribute that was erased, or nullptr if there was no attribute with
    /// such name.
    Attribute removeArgAttr(unsigned index, StringAttr name) {
      return function_interface_impl::removeArgAttr((*static_cast<ConcreteOp *>(this)), index, name);
    }
    Attribute removeArgAttr(unsigned index, StringRef name) {
      return removeArgAttr(
          index, StringAttr::get(this->getOperation()->getContext(), name));
    }

    //===------------------------------------------------------------------===//
    // Result Attributes
    //===------------------------------------------------------------------===//

    /// Return all of the attributes for the result at 'index'.
    ArrayRef<NamedAttribute> getResultAttrs(unsigned index) {
      return function_interface_impl::getResultAttrs(this->getOperation(), index);
    }

    /// Return an ArrayAttr containing all result attribute dictionaries of this
    /// function, or nullptr if no result have attributes.
    ArrayAttr getAllResultAttrs() { return (*static_cast<ConcreteOp *>(this)).getResAttrsAttr(); }

    /// Return all result attributes of this function.
    void getAllResultAttrs(SmallVectorImpl<DictionaryAttr> &result) {
      if (ArrayAttr argAttrs = getAllResultAttrs()) {
        auto argAttrRange = argAttrs.template getAsRange<DictionaryAttr>();
        result.append(argAttrRange.begin(), argAttrRange.end());
      } else {
        result.append((*static_cast<ConcreteOp *>(this)).getNumResults(),
                      DictionaryAttr::get(this->getOperation()->getContext()));
      }
    }

    /// Return the specified attribute, if present, for the result at 'index',
    /// null otherwise.
    Attribute getResultAttr(unsigned index, StringAttr name) {
      auto argDict = getResultAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }
    Attribute getResultAttr(unsigned index, StringRef name) {
      auto argDict = getResultAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }

    template <typename AttrClass>
    AttrClass getResultAttrOfType(unsigned index, StringAttr name) {
      return getResultAttr(index, name).template dyn_cast_or_null<AttrClass>();
    }
    template <typename AttrClass>
    AttrClass getResultAttrOfType(unsigned index, StringRef name) {
      return getResultAttr(index, name).template dyn_cast_or_null<AttrClass>();
    }

    /// Set the attributes held by the result at 'index'.
    void setResultAttrs(unsigned index, ArrayRef<NamedAttribute> attributes) {
      function_interface_impl::setResultAttrs((*static_cast<ConcreteOp *>(this)), index, attributes);
    }

    /// Set the attributes held by the result at 'index'. `attributes` may be
    /// null, in which case any existing argument attributes are removed.
    void setResultAttrs(unsigned index, DictionaryAttr attributes) {
      function_interface_impl::setResultAttrs((*static_cast<ConcreteOp *>(this)), index, attributes);
    }
    void setAllResultAttrs(ArrayRef<DictionaryAttr> attributes) {
      assert(attributes.size() == (*static_cast<ConcreteOp *>(this)).getNumResults());
      function_interface_impl::setAllResultAttrDicts(
        this->getOperation(), attributes);
    }
    void setAllResultAttrs(ArrayRef<Attribute> attributes) {
      assert(attributes.size() == (*static_cast<ConcreteOp *>(this)).getNumResults());
      function_interface_impl::setAllResultAttrDicts(
        this->getOperation(), attributes);
    }
    void setAllResultAttrs(ArrayAttr attributes) {
      assert(attributes.size() == (*static_cast<ConcreteOp *>(this)).getNumResults());
      (*static_cast<ConcreteOp *>(this)).setResAttrsAttr(attributes);
    }

    /// If the an attribute exists with the specified name, change it to the new
    /// value. Otherwise, add a new attribute with the specified name/value.
    void setResultAttr(unsigned index, StringAttr name, Attribute value) {
      function_interface_impl::setResultAttr((*static_cast<ConcreteOp *>(this)), index, name, value);
    }
    void setResultAttr(unsigned index, StringRef name, Attribute value) {
      setResultAttr(index,
                    StringAttr::get(this->getOperation()->getContext(), name),
                    value);
    }

    /// Remove the attribute 'name' from the result at 'index'. Return the
    /// attribute that was erased, or nullptr if there was no attribute with
    /// such name.
    Attribute removeResultAttr(unsigned index, StringAttr name) {
      return function_interface_impl::removeResultAttr((*static_cast<ConcreteOp *>(this)), index, name);
    }

    /// Returns the dictionary attribute corresponding to the argument at
    /// 'index'. If there are no argument attributes at 'index', a null
    /// attribute is returned.
    DictionaryAttr getArgAttrDict(unsigned index) {
      assert(index < (*static_cast<ConcreteOp *>(this)).getNumArguments() && "invalid argument number");
      return function_interface_impl::getArgAttrDict(this->getOperation(), index);
    }

    /// Returns the dictionary attribute corresponding to the result at 'index'.
    /// If there are no result attributes at 'index', a null attribute is
    /// returned.
    DictionaryAttr getResultAttrDict(unsigned index) {
      assert(index < (*static_cast<ConcreteOp *>(this)).getNumResults() && "invalid result number");
      return function_interface_impl::getResultAttrDict(this->getOperation(), index);
    }
  
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::mlir::Type detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::getFunctionType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getFunctionType();
}
template<typename ConcreteOp>
void detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::setFunctionTypeAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeAttr type) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setFunctionTypeAttr(type);
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::getArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getArgAttrsAttr();
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::getResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getResAttrsAttr();
}
template<typename ConcreteOp>
void detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::setArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attrs) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setArgAttrsAttr(attrs);
}
template<typename ConcreteOp>
void detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::setResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attrs) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setResAttrsAttr(attrs);
}
template<typename ConcreteOp>
::mlir::Attribute detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::removeArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).removeArgAttrsAttr();
}
template<typename ConcreteOp>
::mlir::Attribute detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::removeResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).removeResAttrsAttr();
}
template<typename ConcreteOp>
::llvm::ArrayRef<::mlir::Type> detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::getArgumentTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getArgumentTypes();
}
template<typename ConcreteOp>
::llvm::ArrayRef<::mlir::Type> detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::getResultTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getResultTypes();
}
template<typename ConcreteOp>
::mlir::Type detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::cloneTypeWith(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeRange inputs, ::mlir::TypeRange results) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).cloneTypeWith(inputs, results);
}
template<typename ConcreteOp>
::mlir::LogicalResult detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::verifyBody(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).verifyBody();
}
template<typename ConcreteOp>
::mlir::LogicalResult detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::verifyType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).verifyType();
}
template<typename ConcreteOp>
::mlir::Type detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getFunctionType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getFunctionType(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setFunctionTypeAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeAttr type) {
  return static_cast<const ConcreteOp *>(impl)->setFunctionTypeAttr(tablegen_opaque_val, type);
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getArgAttrsAttr(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getResAttrsAttr(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attrs) {
  return static_cast<const ConcreteOp *>(impl)->setArgAttrsAttr(tablegen_opaque_val, attrs);
}
template<typename ConcreteOp>
void detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attrs) {
  return static_cast<const ConcreteOp *>(impl)->setResAttrsAttr(tablegen_opaque_val, attrs);
}
template<typename ConcreteOp>
::mlir::Attribute detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::removeArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->removeArgAttrsAttr(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Attribute detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::removeResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->removeResAttrsAttr(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::ArrayRef<::mlir::Type> detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getArgumentTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getArgumentTypes(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::ArrayRef<::mlir::Type> detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getResultTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getResultTypes(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Type detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::cloneTypeWith(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeRange inputs, ::mlir::TypeRange results) {
  return static_cast<const ConcreteOp *>(impl)->cloneTypeWith(tablegen_opaque_val, inputs, results);
}
template<typename ConcreteOp>
::mlir::LogicalResult detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::verifyBody(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->verifyBody(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::LogicalResult detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::verifyType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->verifyType(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Type detail::FunctionOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::cloneTypeWith(::mlir::Operation *tablegen_opaque_val, ::mlir::TypeRange inputs, ::mlir::TypeRange results) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getFunctionType().clone(inputs, results);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::LogicalResult detail::FunctionOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::verifyBody(::mlir::Operation *tablegen_opaque_val) const {
if ((llvm::cast<ConcreteOp>(tablegen_opaque_val)).isExternal())
        return success();
      ArrayRef<Type> fnInputTypes = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getArgumentTypes();
      // NOTE: This should just be (llvm::cast<ConcreteOp>(tablegen_opaque_val)).front() but access generically
      // because the interface methods defined here may be shadowed in
      // arbitrary ways. https://github.com/llvm/llvm-project/issues/54807
      Block &entryBlock = (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getRegion(0).front();

      unsigned numArguments = fnInputTypes.size();
      if (entryBlock.getNumArguments() != numArguments)
        return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).emitOpError("entry block must have ")
              << numArguments << " arguments to match function signature";

      for (unsigned i = 0, e = fnInputTypes.size(); i != e; ++i) {
        Type argType = entryBlock.getArgument(i).getType();
        if (fnInputTypes[i] != argType) {
          return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).emitOpError("type of entry block argument #")
                << i << '(' << argType
                << ") must match the type of the corresponding argument in "
                << "function signature(" << fnInputTypes[i] << ')';
        }
      }

      return success();
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::LogicalResult detail::FunctionOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::verifyType(::mlir::Operation *tablegen_opaque_val) const {
return success();
}
} // namespace mlir
