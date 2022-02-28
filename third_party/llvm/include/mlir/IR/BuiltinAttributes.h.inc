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
class AffineMapAttr;
class ArrayAttr;
class DenseIntOrFPElementsAttr;
class DenseStringElementsAttr;
class DictionaryAttr;
class FloatAttr;
class IntegerAttr;
class IntegerSetAttr;
class OpaqueAttr;
class OpaqueElementsAttr;
class SparseElementsAttr;
class StringAttr;
class SymbolRefAttr;
class TypeAttr;
class UnitAttr;
namespace detail {
struct AffineMapAttrStorage;
} // namespace detail
class AffineMapAttr : public ::mlir::Attribute::AttrBase<AffineMapAttr, ::mlir::Attribute, detail::AffineMapAttrStorage, ::mlir::MemRefLayoutAttrInterface::Trait> {
public:
  using Base::Base;
  using ValueType = AffineMap;
  AffineMap getAffineMap() const { return getValue(); }
public:
  static AffineMapAttr get(AffineMap value);
  AffineMap getValue() const;
};
namespace detail {
struct ArrayAttrStorage;
} // namespace detail
class ArrayAttr : public ::mlir::Attribute::AttrBase<ArrayAttr, ::mlir::Attribute, detail::ArrayAttrStorage, ::mlir::SubElementAttrInterface::Trait> {
public:
  using Base::Base;
    using ValueType = ArrayRef<Attribute>;

    /// Return the element at the given index.
    Attribute operator[](unsigned idx) const {
      assert(idx < size() && "index out of bounds");
      return getValue()[idx];
    }

    /// Support range iteration.
    using iterator = llvm::ArrayRef<Attribute>::iterator;
    iterator begin() const { return getValue().begin(); }
    iterator end() const { return getValue().end(); }
    size_t size() const { return getValue().size(); }
    bool empty() const { return size() == 0; }

  private:
    /// Class for underlying value iterator support.
    template <typename AttrTy>
    class attr_value_iterator final
        : public llvm::mapped_iterator<ArrayAttr::iterator,
                                       AttrTy (*)(Attribute)> {
    public:
      explicit attr_value_iterator(ArrayAttr::iterator it)
          : llvm::mapped_iterator<ArrayAttr::iterator, AttrTy (*)(Attribute)>(
                it, [](Attribute attr) { return attr.cast<AttrTy>(); }) {}
      AttrTy operator*() const { return (*this->I).template cast<AttrTy>(); }
    };

  public:
    template <typename AttrTy>
    iterator_range<attr_value_iterator<AttrTy>> getAsRange() const {
      return llvm::make_range(attr_value_iterator<AttrTy>(begin()),
                              attr_value_iterator<AttrTy>(end()));
    }
    template <typename AttrTy,
              typename UnderlyingTy = typename AttrTy::ValueType>
    auto getAsValueRange() const {
      return llvm::map_range(getAsRange<AttrTy>(), [](AttrTy attr) {
        return static_cast<UnderlyingTy>(attr.getValue());
      });
    }
public:
  static ArrayAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<Attribute> value);
  ::llvm::ArrayRef<Attribute> getValue() const;
  void walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) const;
  ::mlir::SubElementAttrInterface replaceImmediateSubAttribute(::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const;
};
namespace detail {
struct DenseIntOrFPElementsAttrStorage;
} // namespace detail
class DenseIntOrFPElementsAttr : public ::mlir::Attribute::AttrBase<DenseIntOrFPElementsAttr, DenseElementsAttr, detail::DenseIntOrFPElementsAttrStorage, ::mlir::ElementsAttr::Trait> {
public:
  using Base::Base;
    using DenseElementsAttr::empty;
    using DenseElementsAttr::getNumElements;
    using DenseElementsAttr::getElementType;
    using DenseElementsAttr::getValues;
    using DenseElementsAttr::isSplat;
    using DenseElementsAttr::size;
    using DenseElementsAttr::value_begin;

    /// The set of data types that can be iterated by this attribute.
    using ContiguousIterableTypesT = std::tuple<
      // Integer types.
      uint8_t, uint16_t, uint32_t, uint64_t,
      int8_t, int16_t, int32_t, int64_t,
      short, unsigned short, int, unsigned, long, unsigned long,
      std::complex<uint8_t>, std::complex<uint16_t>, std::complex<uint32_t>,
      std::complex<uint64_t>,
      std::complex<int8_t>, std::complex<int16_t>, std::complex<int32_t>,
      std::complex<int64_t>,
      // Float types.
      float, double, std::complex<float>, std::complex<double>
    >;
    using NonContiguousIterableTypesT = std::tuple<
      Attribute,
      // Integer types.
      APInt, bool, std::complex<APInt>,
      // Float types.
      APFloat, std::complex<APFloat>
    >;

    /// Provide a `value_begin_impl` to enable iteration within ElementsAttr.
    template <typename T>
    auto value_begin_impl(OverloadToken<T>) const {
      return value_begin<T>();
    }

    /// Convert endianess of input ArrayRef for big-endian(BE) machines. All of
    /// the elements of `inRawData` has `type`. If `inRawData` is little endian
    /// (LE), it is converted to big endian (BE). Conversely, if `inRawData` is
    /// BE, converted to LE.
    static void
    convertEndianOfArrayRefForBEmachine(ArrayRef<char> inRawData,
                                        MutableArrayRef<char> outRawData,
                                        ShapedType type);

    /// Convert endianess of input for big-endian(BE) machines. The number of
    /// elements of `inRawData` is `numElements`, and each element has
    /// `elementBitWidth` bits. If `inRawData` is little endian (LE), it is
    /// converted to big endian (BE) and saved in `outRawData`. Conversely, if
    /// `inRawData` is BE, converted to LE.
    static void convertEndianOfCharForBEmachine(const char *inRawData,
                                                char *outRawData,
                                                size_t elementBitWidth,
                                                size_t numElements);

  protected:
    friend DenseElementsAttr;

    /// Constructs a dense elements attribute from an array of raw APFloat
    /// values. Each APFloat value is expected to have the same bitwidth as the
    /// element type of 'type'. 'type' must be a vector or tensor with static
    /// shape.
    static DenseElementsAttr getRaw(ShapedType type, size_t storageWidth,
                                    ArrayRef<APFloat> values, bool isSplat);

    /// Constructs a dense elements attribute from an array of raw APInt values.
    /// Each APInt value is expected to have the same bitwidth as the element
    /// type of 'type'. 'type' must be a vector or tensor with static shape.
    static DenseElementsAttr getRaw(ShapedType type, size_t storageWidth,
                                    ArrayRef<APInt> values, bool isSplat);

    /// Get or create a new dense elements attribute instance with the given raw
    /// data buffer. 'type' must be a vector or tensor with static shape.
    static DenseElementsAttr getRaw(ShapedType type, ArrayRef<char> data,
                                    bool isSplat);

    /// Overload of the raw 'get' method that asserts that the given type is of
    /// complex type. This method is used to verify type invariants that the
    /// templatized 'get' method cannot.
    static DenseElementsAttr getRawComplex(ShapedType type, ArrayRef<char> data,
                                           int64_t dataEltSize, bool isInt,
                                           bool isSigned);

    /// Overload of the raw 'get' method that asserts that the given type is of
    /// integer or floating-point type. This method is used to verify type
    /// invariants that the templatized 'get' method cannot.
    static DenseElementsAttr getRawIntOrFloat(ShapedType type,
                                              ArrayRef<char> data,
                                              int64_t dataEltSize, bool isInt,
                                              bool isSigned);

  public:
};
namespace detail {
struct DenseStringElementsAttrStorage;
} // namespace detail
class DenseStringElementsAttr : public ::mlir::Attribute::AttrBase<DenseStringElementsAttr, DenseElementsAttr, detail::DenseStringElementsAttrStorage, ::mlir::ElementsAttr::Trait> {
public:
  using Base::Base;
    using DenseElementsAttr::empty;
    using DenseElementsAttr::getNumElements;
    using DenseElementsAttr::getElementType;
    using DenseElementsAttr::getValues;
    using DenseElementsAttr::isSplat;
    using DenseElementsAttr::size;
    using DenseElementsAttr::value_begin;

    /// The set of data types that can be iterated by this attribute.
    using ContiguousIterableTypesT = std::tuple<StringRef>;
    using NonContiguousIterableTypesT = std::tuple<Attribute>;

    /// Provide a `value_begin_impl` to enable iteration within ElementsAttr.
    template <typename T>
    auto value_begin_impl(OverloadToken<T>) const {
      return value_begin<T>();
    }

  protected:
    friend DenseElementsAttr;

  public:
public:
  static DenseStringElementsAttr get(ShapedType type, ArrayRef<StringRef> values);
};
namespace detail {
struct DictionaryAttrStorage;
} // namespace detail
class DictionaryAttr : public ::mlir::Attribute::AttrBase<DictionaryAttr, ::mlir::Attribute, detail::DictionaryAttrStorage, ::mlir::SubElementAttrInterface::Trait> {
public:
  using Base::Base;
    using ValueType = ArrayRef<NamedAttribute>;

    /// Construct a dictionary with an array of values that is known to already
    /// be sorted by name and uniqued.
    static DictionaryAttr getWithSorted(MLIRContext *context,
                                        ArrayRef<NamedAttribute> value);

    /// Return the specified attribute if present, null otherwise.
    Attribute get(StringRef name) const;
    Attribute get(StringAttr name) const;

    /// Return the specified named attribute if present, None otherwise.
    Optional<NamedAttribute> getNamed(StringRef name) const;
    Optional<NamedAttribute> getNamed(StringAttr name) const;

    /// Return whether the specified attribute is present.
    bool contains(StringRef name) const;
    bool contains(StringAttr name) const;

    /// Support range iteration.
    using iterator = llvm::ArrayRef<NamedAttribute>::iterator;
    iterator begin() const;
    iterator end() const;
    bool empty() const { return size() == 0; }
    size_t size() const;

    /// Sorts the NamedAttributes in the array ordered by name as expected by
    /// getWithSorted and returns whether the values were sorted.
    /// Requires: uniquely named attributes.
    static bool sort(ArrayRef<NamedAttribute> values,
                     SmallVectorImpl<NamedAttribute> &storage);

    /// Sorts the NamedAttributes in the array ordered by name as expected by
    /// getWithSorted in place on an array and returns whether the values needed
    /// to be sorted.
    /// Requires: uniquely named attributes.
    static bool sortInPlace(SmallVectorImpl<NamedAttribute> &array);

    /// Returns an entry with a duplicate name in `array`, if it exists, else
    /// returns llvm::None. If `isSorted` is true, the array is assumed to be
    /// sorted else it will be sorted in place before finding the duplicate entry.
    static Optional<NamedAttribute>
    findDuplicate(SmallVectorImpl<NamedAttribute> &array, bool isSorted);

    /// Return the specified attribute if present and is an instance of
    /// `AttrClass`, null otherwise.
    template<typename AttrClass, typename NameClass>
    AttrClass getAs(NameClass &&name) const {
      return get(std::forward<NameClass>(name))
        .template dyn_cast_or_null<AttrClass>();
    }

  private:
    /// Return empty dictionary.
    static DictionaryAttr getEmpty(MLIRContext *context);

    /// Return empty dictionary. This is a special variant of the above method
    /// that is used by the MLIRContext to cache the empty dictionary instance.
    static DictionaryAttr getEmptyUnchecked(MLIRContext *context);

    /// Allow access to `getEmptyUnchecked`.
    friend MLIRContext;

  public:
public:
  static DictionaryAttr get(::mlir::MLIRContext *context, ArrayRef<NamedAttribute> value = llvm::None);
  ::llvm::ArrayRef<NamedAttribute> getValue() const;
  void walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) const;
  ::mlir::SubElementAttrInterface replaceImmediateSubAttribute(::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const;
};
namespace detail {
struct FloatAttrStorage;
} // namespace detail
class FloatAttr : public ::mlir::Attribute::AttrBase<FloatAttr, ::mlir::Attribute, detail::FloatAttrStorage> {
public:
  using Base::Base;
  using ValueType = APFloat;

  /// This function is used to convert the value to a double, even if it loses
  /// precision.
  double getValueAsDouble() const;
  static double getValueAsDouble(APFloat val);
  using Base::getChecked;
public:
  static FloatAttr get(Type type, const APFloat &value);
  static FloatAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type type, const APFloat &value);
  static FloatAttr get(Type type, double value);
  static FloatAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type type, double value);
  static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::Type type, ::llvm::APFloat value);
  ::mlir::Type getType() const;
  ::llvm::APFloat getValue() const;
};
namespace detail {
struct IntegerAttrStorage;
} // namespace detail
class IntegerAttr : public ::mlir::Attribute::AttrBase<IntegerAttr, ::mlir::Attribute, detail::IntegerAttrStorage> {
public:
  using Base::Base;
    using ValueType = APInt;

    /// Return the integer value as a 64-bit int. The attribute must be a
    /// signless integer.
    // TODO: Change callers to use getValue instead.
    int64_t getInt() const;
    /// Return the integer value as a signed 64-bit int. The attribute must be
    /// a signed integer.
    int64_t getSInt() const;
    /// Return the integer value as a unsigned 64-bit int. The attribute must be
    /// an unsigned integer.
    uint64_t getUInt() const;

    /// Return the value as an APSInt which carries the signed from the type of
    /// the attribute.  This traps on signless integers types!
    APSInt getAPSInt() const;

  private:
    /// Return a boolean attribute. This is a special variant of the `get`
    /// method that is used by the MLIRContext to cache the boolean IntegerAttr
    /// instances.
    static BoolAttr getBoolAttrUnchecked(IntegerType type, bool value);

    /// Allow access to `getBoolAttrUnchecked`.
    friend MLIRContext;

  public:
  using Base::getChecked;
public:
  static IntegerAttr get(Type type, const APInt &value);
  static IntegerAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type type, const APInt &value);
  static IntegerAttr get(::mlir::MLIRContext *context, const APSInt &value);
  static IntegerAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, const APSInt &value);
  static IntegerAttr get(Type type, int64_t value);
  static IntegerAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type type, int64_t value);
  static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::Type type, APInt value);
  ::mlir::Type getType() const;
  APInt getValue() const;
};
namespace detail {
struct IntegerSetAttrStorage;
} // namespace detail
class IntegerSetAttr : public ::mlir::Attribute::AttrBase<IntegerSetAttr, ::mlir::Attribute, detail::IntegerSetAttrStorage> {
public:
  using Base::Base;
  using ValueType = IntegerSet;public:
  static IntegerSetAttr get(IntegerSet value);
  IntegerSet getValue() const;
};
namespace detail {
struct OpaqueAttrStorage;
} // namespace detail
class OpaqueAttr : public ::mlir::Attribute::AttrBase<OpaqueAttr, ::mlir::Attribute, detail::OpaqueAttrStorage> {
public:
  using Base::Base;
  using Base::getChecked;
public:
  static OpaqueAttr get(StringAttr dialect, StringRef attrData, Type type);
  static OpaqueAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, StringAttr dialect, StringRef attrData, Type type);
  static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, StringAttr dialectNamespace, ::llvm::StringRef attrData, ::mlir::Type type);
  StringAttr getDialectNamespace() const;
  ::llvm::StringRef getAttrData() const;
  ::mlir::Type getType() const;
};
namespace detail {
struct OpaqueElementsAttrStorage;
} // namespace detail
class OpaqueElementsAttr : public ::mlir::Attribute::AttrBase<OpaqueElementsAttr, ::mlir::Attribute, detail::OpaqueElementsAttrStorage, ::mlir::ElementsAttr::Trait> {
public:
  using Base::Base;
  using ValueType = StringRef;

  /// Decodes the attribute value using dialect-specific decoding hook.
  /// Returns false if decoding is successful. If not, returns true and leaves
  /// 'result' argument unspecified.
  bool decode(ElementsAttr &result);
  using Base::getChecked;
public:
  static OpaqueElementsAttr get(StringAttr dialect, ShapedType type, StringRef value);
  static OpaqueElementsAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, StringAttr dialect, ShapedType type, StringRef value);
  static OpaqueElementsAttr get(Dialect *dialect, ShapedType type, StringRef value);
  static OpaqueElementsAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Dialect *dialect, ShapedType type, StringRef value);
  static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, StringAttr dialect, ::llvm::StringRef value, ShapedType type);
  StringAttr getDialect() const;
  ::llvm::StringRef getValue() const;
  ShapedType getType() const;
};
namespace detail {
struct SparseElementsAttrStorage;
} // namespace detail
class SparseElementsAttr : public ::mlir::Attribute::AttrBase<SparseElementsAttr, ::mlir::Attribute, detail::SparseElementsAttrStorage, ::mlir::ElementsAttr::Trait> {
public:
  using Base::Base;
    /// The set of data types that can be iterated by this attribute.
    // FIXME: Realistically, SparseElementsAttr could use ElementsAttr for the
    // value storage. This would mean dispatching to `values` when accessing
    // values. For now, we just add the types that can be iterated by
    // DenseElementsAttr.
    using NonContiguousIterableTypesT = std::tuple<
      Attribute,
      // Integer types.
      APInt, bool, uint8_t, uint16_t, uint32_t, uint64_t,
      int8_t, int16_t, int32_t, int64_t,
      short, unsigned short, int, unsigned, long, unsigned long,
      std::complex<APInt>, std::complex<uint8_t>, std::complex<uint16_t>,
      std::complex<uint32_t>, std::complex<uint64_t>, std::complex<int8_t>,
      std::complex<int16_t>, std::complex<int32_t>, std::complex<int64_t>,
      // Float types.
      APFloat, float, double,
      std::complex<APFloat>, std::complex<float>, std::complex<double>,
      // String types.
      StringRef
    >;
    using ElementsAttr::Trait<SparseElementsAttr>::getValues;

    /// Provide a `value_begin_impl` to enable iteration within ElementsAttr.
    template <typename T>
    auto value_begin_impl(OverloadToken<T>) const {
      return value_begin<T>();
    }

    template <typename T>
    using iterator =
        llvm::mapped_iterator<typename decltype(llvm::seq<ptrdiff_t>(0, 0))::iterator,
                              std::function<T(ptrdiff_t)>>;

    /// Return the values of this attribute in the form of the given type 'T'.
    /// 'T'  may be any of Attribute, APInt, APFloat, c++ integer/float types,
    /// etc.
    template <typename T> iterator<T> value_begin() const;

  private:
    /// Get a zero APFloat for the given sparse attribute.
    APFloat getZeroAPFloat() const;

    /// Get a zero APInt for the given sparse attribute.
    APInt getZeroAPInt() const;

    /// Get a zero attribute for the given sparse attribute.
    Attribute getZeroAttr() const;

    /// Utility methods to generate a zero value of some type 'T'. This is used
    /// by the 'iterator' class.
    /// Get a zero for a given attribute type.
    template <typename T>
    typename std::enable_if<std::is_base_of<Attribute, T>::value, T>::type
    getZeroValue() const {
      return getZeroAttr().template cast<T>();
    }
    /// Get a zero for an APInt.
    template <typename T>
    typename std::enable_if<std::is_same<APInt, T>::value, T>::type
    getZeroValue() const {
      return getZeroAPInt();
    }
    template <typename T>
    typename std::enable_if<std::is_same<std::complex<APInt>, T>::value,
                            T>::type
    getZeroValue() const {
      APInt intZero = getZeroAPInt();
      return {intZero, intZero};
    }
    /// Get a zero for an APFloat.
    template <typename T>
    typename std::enable_if<std::is_same<APFloat, T>::value, T>::type
    getZeroValue() const {
      return getZeroAPFloat();
    }
    template <typename T>
    typename std::enable_if<std::is_same<std::complex<APFloat>, T>::value,
                            T>::type
    getZeroValue() const {
      APFloat floatZero = getZeroAPFloat();
      return {floatZero, floatZero};
    }

    /// Get a zero for an C++ integer, float, StringRef, or complex type.
    template <typename T>
    typename std::enable_if<
        std::numeric_limits<T>::is_integer ||
            DenseElementsAttr::is_valid_cpp_fp_type<T>::value ||
            std::is_same<T, StringRef>::value ||
            (detail::is_complex_t<T>::value &&
             !llvm::is_one_of<T, std::complex<APInt>,
                              std::complex<APFloat>>::value),
        T>::type
    getZeroValue() const {
      return T();
    }

    /// Flatten, and return, all of the sparse indices in this attribute in
    /// row-major order.
    std::vector<ptrdiff_t> getFlattenedSparseIndices() const;

  public:
  using Base::getChecked;
public:
  static SparseElementsAttr get(ShapedType type, DenseElementsAttr indices, DenseElementsAttr values);
  static SparseElementsAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ShapedType type, DenseElementsAttr indices, DenseElementsAttr values);
  static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ShapedType type, DenseIntElementsAttr indices, DenseElementsAttr values);
  ShapedType getType() const;
  DenseIntElementsAttr getIndices() const;
  DenseElementsAttr getValues() const;
};
namespace detail {
struct StringAttrStorage;
} // namespace detail
class StringAttr : public ::mlir::Attribute::AttrBase<StringAttr, ::mlir::Attribute, detail::StringAttrStorage> {
public:
  using Base::Base;
    using ValueType = StringRef;

    /// If the value of this string is prefixed with a dialect namespace,
    /// returns the dialect corresponding to that namespace if it is loaded,
    /// nullptr otherwise. For example, the string `llvm.fastmathflags` would
    /// return the LLVM dialect, assuming it is loaded in the context.
    Dialect *getReferencedDialect() const;

    /// Enable conversion to StringRef.
    operator StringRef() const { return getValue(); }

    /// Returns the underlying string value
    StringRef strref() const { return getValue(); }

    /// Convert the underling value to an std::string.
    std::string str() const { return getValue().str(); }

    /// Return a pointer to the start of the string data.
    const char *data() const { return getValue().data(); }

    /// Return the number of bytes in this string.
    size_t size() const { return getValue().size(); }

    /// Iterate over the underlying string data.
    StringRef::iterator begin() const { return getValue().begin(); }
    StringRef::iterator end() const { return getValue().end(); }

    /// Compare the underlying string value to the one in `rhs`.
    int compare(StringAttr rhs) const {
      if (*this == rhs)
        return 0;
      return getValue().compare(rhs.getValue());
    }

  private:
    /// Return an empty StringAttr with NoneType type. This is a special variant
    /// of the `get` method that is used by the MLIRContext to cache the
    /// instance.
    static StringAttr getEmptyStringAttrUnchecked(MLIRContext *context);
    friend MLIRContext;
  public:
public:
  static StringAttr get(const Twine &bytes, Type type);
  static StringAttr get(::mlir::MLIRContext *context, const Twine &bytes);
  static StringAttr get(::mlir::MLIRContext *context);
  ::llvm::StringRef getValue() const;
  ::mlir::Type getType() const;
};
namespace detail {
struct SymbolRefAttrStorage;
} // namespace detail
class SymbolRefAttr : public ::mlir::Attribute::AttrBase<SymbolRefAttr, ::mlir::Attribute, detail::SymbolRefAttrStorage> {
public:
  using Base::Base;
  static SymbolRefAttr get(MLIRContext *ctx, StringRef value,
                           ArrayRef<FlatSymbolRefAttr> nestedRefs);
  /// Convenience getters for building a SymbolRefAttr with no path, which is
  /// known to produce a FlatSymbolRefAttr.
  static FlatSymbolRefAttr get(StringAttr value);
  static FlatSymbolRefAttr get(MLIRContext *ctx, StringRef value);

  /// Convenience getter for buliding a SymbolRefAttr based on an operation
  /// that implements the SymbolTrait.
  static FlatSymbolRefAttr get(Operation *symbol);

  /// Returns the name of the fully resolved symbol, i.e. the leaf of the
  /// reference path.
  StringAttr getLeafReference() const;
public:
  static SymbolRefAttr get(StringAttr rootReference, ArrayRef<FlatSymbolRefAttr> nestedReferences);
  StringAttr getRootReference() const;
  ::llvm::ArrayRef<FlatSymbolRefAttr> getNestedReferences() const;
};
namespace detail {
struct TypeAttrStorage;
} // namespace detail
class TypeAttr : public ::mlir::Attribute::AttrBase<TypeAttr, ::mlir::Attribute, detail::TypeAttrStorage, ::mlir::SubElementAttrInterface::Trait> {
public:
  using Base::Base;
  using ValueType = Type;public:
  static TypeAttr get(Type type);
  Type getValue() const;
  void walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) const;
};
class UnitAttr : public ::mlir::Attribute::AttrBase<UnitAttr, ::mlir::Attribute, ::mlir::AttributeStorage> {
public:
  using Base::Base;
  static UnitAttr get(MLIRContext *context);
};
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::AffineMapAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::ArrayAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::DenseIntOrFPElementsAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::DenseStringElementsAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::DictionaryAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::FloatAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::IntegerAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::IntegerSetAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::OpaqueAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::OpaqueElementsAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::SparseElementsAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::StringAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::SymbolRefAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::TypeAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::UnitAttr)

#endif  // GET_ATTRDEF_CLASSES

