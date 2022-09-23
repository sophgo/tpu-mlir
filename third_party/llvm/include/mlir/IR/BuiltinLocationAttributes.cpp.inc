/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::CallSiteLoc,
::mlir::FileLineColLoc,
::mlir::FusedLoc,
::mlir::NameLoc,
::mlir::OpaqueLoc,
::mlir::UnknownLoc

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

namespace mlir {
namespace detail {
struct CallSiteLocAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<Location, Location>;
  CallSiteLocAttrStorage(Location callee, Location caller) : callee(callee), caller(caller) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (callee == std::get<0>(tblgenKey)) && (caller == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static CallSiteLocAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto callee = std::get<0>(tblgenKey);
    auto caller = std::get<1>(tblgenKey);
    return new (allocator.allocate<CallSiteLocAttrStorage>()) CallSiteLocAttrStorage(callee, caller);
  }

  Location callee;
  Location caller;
};
} // namespace detail
CallSiteLoc CallSiteLoc::get(Location callee, Location caller) {
  return Base::get(callee->getContext(), callee, caller);
}

Location CallSiteLoc::getCallee() const {
  return getImpl()->callee;
}

Location CallSiteLoc::getCaller() const {
  return getImpl()->caller;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::CallSiteLoc)
namespace mlir {
namespace detail {
struct FileLineColLocAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<StringAttr, unsigned, unsigned>;
  FileLineColLocAttrStorage(StringAttr filename, unsigned line, unsigned column) : filename(filename), line(line), column(column) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (filename == std::get<0>(tblgenKey)) && (line == std::get<1>(tblgenKey)) && (column == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static FileLineColLocAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto filename = std::get<0>(tblgenKey);
    auto line = std::get<1>(tblgenKey);
    auto column = std::get<2>(tblgenKey);
    return new (allocator.allocate<FileLineColLocAttrStorage>()) FileLineColLocAttrStorage(filename, line, column);
  }

  StringAttr filename;
  unsigned line;
  unsigned column;
};
} // namespace detail
FileLineColLoc FileLineColLoc::get(StringAttr filename, unsigned line, unsigned column) {
  return Base::get(filename.getContext(), filename, line, column);
}

FileLineColLoc FileLineColLoc::get(::mlir::MLIRContext *context, StringRef filename, unsigned line, unsigned column) {
  return Base::get(context,
               StringAttr::get(context, filename.empty() ? "-" : filename),
               line, column);
}

StringAttr FileLineColLoc::getFilename() const {
  return getImpl()->filename;
}

unsigned FileLineColLoc::getLine() const {
  return getImpl()->line;
}

unsigned FileLineColLoc::getColumn() const {
  return getImpl()->column;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::FileLineColLoc)
namespace mlir {
namespace detail {
struct FusedLocAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<Location>, Attribute>;
  FusedLocAttrStorage(::llvm::ArrayRef<Location> locations, Attribute metadata) : locations(locations), metadata(metadata) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (locations == std::get<0>(tblgenKey)) && (metadata == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static FusedLocAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto locations = std::get<0>(tblgenKey);
    auto metadata = std::get<1>(tblgenKey);
    locations = allocator.copyInto(locations);
    return new (allocator.allocate<FusedLocAttrStorage>()) FusedLocAttrStorage(locations, metadata);
  }

  ::llvm::ArrayRef<Location> locations;
  Attribute metadata;
};
} // namespace detail
FusedLoc FusedLoc::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<Location> locations, Attribute metadata) {
  return Base::get(context, locations, metadata);
}

::llvm::ArrayRef<Location> FusedLoc::getLocations() const {
  return getImpl()->locations;
}

Attribute FusedLoc::getMetadata() const {
  return getImpl()->metadata;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::FusedLoc)
namespace mlir {
namespace detail {
struct NameLocAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<StringAttr, Location>;
  NameLocAttrStorage(StringAttr name, Location childLoc) : name(name), childLoc(childLoc) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (name == std::get<0>(tblgenKey)) && (childLoc == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static NameLocAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto name = std::get<0>(tblgenKey);
    auto childLoc = std::get<1>(tblgenKey);
    return new (allocator.allocate<NameLocAttrStorage>()) NameLocAttrStorage(name, childLoc);
  }

  StringAttr name;
  Location childLoc;
};
} // namespace detail
NameLoc NameLoc::get(StringAttr name, Location childLoc) {
  return Base::get(name.getContext(), name, childLoc);
}

NameLoc NameLoc::get(StringAttr name) {
  return Base::get(name.getContext(), name,
               UnknownLoc::get(name.getContext()));
}

StringAttr NameLoc::getName() const {
  return getImpl()->name;
}

Location NameLoc::getChildLoc() const {
  return getImpl()->childLoc;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NameLoc)
namespace mlir {
namespace detail {
struct OpaqueLocAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<uintptr_t, TypeID, Location>;
  OpaqueLocAttrStorage(uintptr_t underlyingLocation, TypeID underlyingTypeID, Location fallbackLocation) : underlyingLocation(underlyingLocation), underlyingTypeID(underlyingTypeID), fallbackLocation(fallbackLocation) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (underlyingLocation == std::get<0>(tblgenKey)) && (underlyingTypeID == std::get<1>(tblgenKey)) && (fallbackLocation == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static OpaqueLocAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto underlyingLocation = std::get<0>(tblgenKey);
    auto underlyingTypeID = std::get<1>(tblgenKey);
    auto fallbackLocation = std::get<2>(tblgenKey);
    return new (allocator.allocate<OpaqueLocAttrStorage>()) OpaqueLocAttrStorage(underlyingLocation, underlyingTypeID, fallbackLocation);
  }

  uintptr_t underlyingLocation;
  TypeID underlyingTypeID;
  Location fallbackLocation;
};
} // namespace detail
OpaqueLoc OpaqueLoc::get(uintptr_t underlyingLocation, TypeID underlyingTypeID, Location fallbackLocation) {
  return Base::get(fallbackLocation->getContext(), underlyingLocation,
               underlyingTypeID, fallbackLocation);
}

uintptr_t OpaqueLoc::getUnderlyingLocation() const {
  return getImpl()->underlyingLocation;
}

TypeID OpaqueLoc::getUnderlyingTypeID() const {
  return getImpl()->underlyingTypeID;
}

Location OpaqueLoc::getFallbackLocation() const {
  return getImpl()->fallbackLocation;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::OpaqueLoc)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::UnknownLoc)

#endif  // GET_ATTRDEF_CLASSES

