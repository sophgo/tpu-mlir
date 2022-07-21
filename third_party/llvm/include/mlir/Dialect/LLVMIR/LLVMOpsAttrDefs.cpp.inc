/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::LLVM::CConvAttr,
::mlir::LLVM::FMFAttr,
::mlir::LLVM::LinkageAttr,
::mlir::LLVM::LoopOptionsAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  if (mnemonic == ::mlir::LLVM::CConvAttr::getMnemonic()) {
    value = ::mlir::LLVM::CConvAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::LLVM::FMFAttr::getMnemonic()) {
    value = ::mlir::LLVM::FMFAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::LLVM::LinkageAttr::getMnemonic()) {
    value = ::mlir::LLVM::LinkageAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::LLVM::LoopOptionsAttr::getMnemonic()) {
    value = ::mlir::LLVM::LoopOptionsAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  return {};
}

static ::mlir::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)    .Case<::mlir::LLVM::CConvAttr>([&](auto t) {
      printer << ::mlir::LLVM::CConvAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::LLVM::FMFAttr>([&](auto t) {
      printer << ::mlir::LLVM::FMFAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::LLVM::LinkageAttr>([&](auto t) {
      printer << ::mlir::LLVM::LinkageAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::LLVM::LoopOptionsAttr>([&](auto t) {
      printer << ::mlir::LLVM::LoopOptionsAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace LLVM {
namespace detail {
struct CConvAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<CConv>;
  CConvAttrStorage(CConv CallingConv) : ::mlir::AttributeStorage(), CallingConv(CallingConv) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (CallingConv == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static CConvAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto CallingConv = std::get<0>(tblgenKey);
    return new (allocator.allocate<CConvAttrStorage>()) CConvAttrStorage(CallingConv);
  }

  CConv CallingConv;
};
} // namespace detail
CConvAttr CConvAttr::get(::mlir::MLIRContext *context, CConv CallingConv) {
  return Base::get(context, CallingConv);
}

CConv CConvAttr::getCallingConv() const {
  return getImpl()->CallingConv;
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::CConvAttr)
namespace mlir {
namespace LLVM {
namespace detail {
struct FMFAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<FastmathFlags>;
  FMFAttrStorage(FastmathFlags flags) : ::mlir::AttributeStorage(), flags(flags) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (flags == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static FMFAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto flags = std::get<0>(tblgenKey);
    return new (allocator.allocate<FMFAttrStorage>()) FMFAttrStorage(flags);
  }

  FastmathFlags flags;
};
} // namespace detail
FMFAttr FMFAttr::get(::mlir::MLIRContext *context, FastmathFlags flags) {
  return Base::get(context, flags);
}

FastmathFlags FMFAttr::getFlags() const {
  return getImpl()->flags;
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FMFAttr)
namespace mlir {
namespace LLVM {
namespace detail {
struct LinkageAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<linkage::Linkage>;
  LinkageAttrStorage(linkage::Linkage linkage) : ::mlir::AttributeStorage(), linkage(linkage) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (linkage == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static LinkageAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto linkage = std::get<0>(tblgenKey);
    return new (allocator.allocate<LinkageAttrStorage>()) LinkageAttrStorage(linkage);
  }

  linkage::Linkage linkage;
};
} // namespace detail
LinkageAttr LinkageAttr::get(::mlir::MLIRContext *context, linkage::Linkage linkage) {
  return Base::get(context, linkage);
}

linkage::Linkage LinkageAttr::getLinkage() const {
  return getImpl()->linkage;
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LinkageAttr)
namespace mlir {
namespace LLVM {
namespace detail {
struct LoopOptionsAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<std::pair<LoopOptionCase, int64_t>>>;
  LoopOptionsAttrStorage(::llvm::ArrayRef<std::pair<LoopOptionCase, int64_t>> options) : ::mlir::AttributeStorage(), options(options) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (options == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static LoopOptionsAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto options = std::get<0>(tblgenKey);
    options = allocator.copyInto(options);
    return new (allocator.allocate<LoopOptionsAttrStorage>()) LoopOptionsAttrStorage(options);
  }

  ::llvm::ArrayRef<std::pair<LoopOptionCase, int64_t>> options;
};
} // namespace detail
::llvm::ArrayRef<std::pair<LoopOptionCase, int64_t>> LoopOptionsAttr::getOptions() const {
  return getImpl()->options;
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LoopOptionsAttr)
namespace mlir {
namespace LLVM {

/// Parse an attribute registered to this dialect.
::mlir::Attribute LLVMDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                      ::mlir::Type type) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef attrTag;
  if (::mlir::failed(parser.parseKeyword(&attrTag)))
    return {};
  {
    ::mlir::Attribute attr;
    auto parseResult = generatedAttributeParser(parser, attrTag, type, attr);
    if (parseResult.hasValue())
      return attr;
  }
  
  parser.emitError(typeLoc) << "unknown attribute `"
      << attrTag << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print an attribute registered to this dialect.
void LLVMDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace LLVM
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

