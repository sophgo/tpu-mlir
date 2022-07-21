#ifdef GET_ATTR_NAMES
#undef GET_ATTR_NAMES
#ifndef ATTRIBUTE_ALL
#define ATTRIBUTE_ALL(FIRST, SECOND)
#endif

#ifndef ATTRIBUTE_ENUM
#define ATTRIBUTE_ENUM(FIRST, SECOND) ATTRIBUTE_ALL(FIRST, SECOND)
#endif

ATTRIBUTE_ENUM(AllocAlign,allocalign)
ATTRIBUTE_ENUM(AllocatedPointer,allocptr)
ATTRIBUTE_ENUM(AlwaysInline,alwaysinline)
ATTRIBUTE_ENUM(ArgMemOnly,argmemonly)
ATTRIBUTE_ENUM(Builtin,builtin)
ATTRIBUTE_ENUM(Cold,cold)
ATTRIBUTE_ENUM(Convergent,convergent)
ATTRIBUTE_ENUM(DisableSanitizerInstrumentation,disable_sanitizer_instrumentation)
ATTRIBUTE_ENUM(Hot,hot)
ATTRIBUTE_ENUM(ImmArg,immarg)
ATTRIBUTE_ENUM(InReg,inreg)
ATTRIBUTE_ENUM(InaccessibleMemOnly,inaccessiblememonly)
ATTRIBUTE_ENUM(InaccessibleMemOrArgMemOnly,inaccessiblemem_or_argmemonly)
ATTRIBUTE_ENUM(InlineHint,inlinehint)
ATTRIBUTE_ENUM(JumpTable,jumptable)
ATTRIBUTE_ENUM(MinSize,minsize)
ATTRIBUTE_ENUM(MustProgress,mustprogress)
ATTRIBUTE_ENUM(Naked,naked)
ATTRIBUTE_ENUM(Nest,nest)
ATTRIBUTE_ENUM(NoAlias,noalias)
ATTRIBUTE_ENUM(NoBuiltin,nobuiltin)
ATTRIBUTE_ENUM(NoCallback,nocallback)
ATTRIBUTE_ENUM(NoCapture,nocapture)
ATTRIBUTE_ENUM(NoCfCheck,nocf_check)
ATTRIBUTE_ENUM(NoDuplicate,noduplicate)
ATTRIBUTE_ENUM(NoFree,nofree)
ATTRIBUTE_ENUM(NoImplicitFloat,noimplicitfloat)
ATTRIBUTE_ENUM(NoInline,noinline)
ATTRIBUTE_ENUM(NoMerge,nomerge)
ATTRIBUTE_ENUM(NoProfile,noprofile)
ATTRIBUTE_ENUM(NoRecurse,norecurse)
ATTRIBUTE_ENUM(NoRedZone,noredzone)
ATTRIBUTE_ENUM(NoReturn,noreturn)
ATTRIBUTE_ENUM(NoSanitizeBounds,nosanitize_bounds)
ATTRIBUTE_ENUM(NoSanitizeCoverage,nosanitize_coverage)
ATTRIBUTE_ENUM(NoSync,nosync)
ATTRIBUTE_ENUM(NoUndef,noundef)
ATTRIBUTE_ENUM(NoUnwind,nounwind)
ATTRIBUTE_ENUM(NonLazyBind,nonlazybind)
ATTRIBUTE_ENUM(NonNull,nonnull)
ATTRIBUTE_ENUM(NullPointerIsValid,null_pointer_is_valid)
ATTRIBUTE_ENUM(OptForFuzzing,optforfuzzing)
ATTRIBUTE_ENUM(OptimizeForSize,optsize)
ATTRIBUTE_ENUM(OptimizeNone,optnone)
ATTRIBUTE_ENUM(ReadNone,readnone)
ATTRIBUTE_ENUM(ReadOnly,readonly)
ATTRIBUTE_ENUM(Returned,returned)
ATTRIBUTE_ENUM(ReturnsTwice,returns_twice)
ATTRIBUTE_ENUM(SExt,signext)
ATTRIBUTE_ENUM(SafeStack,safestack)
ATTRIBUTE_ENUM(SanitizeAddress,sanitize_address)
ATTRIBUTE_ENUM(SanitizeHWAddress,sanitize_hwaddress)
ATTRIBUTE_ENUM(SanitizeMemTag,sanitize_memtag)
ATTRIBUTE_ENUM(SanitizeMemory,sanitize_memory)
ATTRIBUTE_ENUM(SanitizeThread,sanitize_thread)
ATTRIBUTE_ENUM(ShadowCallStack,shadowcallstack)
ATTRIBUTE_ENUM(Speculatable,speculatable)
ATTRIBUTE_ENUM(SpeculativeLoadHardening,speculative_load_hardening)
ATTRIBUTE_ENUM(StackProtect,ssp)
ATTRIBUTE_ENUM(StackProtectReq,sspreq)
ATTRIBUTE_ENUM(StackProtectStrong,sspstrong)
ATTRIBUTE_ENUM(StrictFP,strictfp)
ATTRIBUTE_ENUM(SwiftAsync,swiftasync)
ATTRIBUTE_ENUM(SwiftError,swifterror)
ATTRIBUTE_ENUM(SwiftSelf,swiftself)
ATTRIBUTE_ENUM(WillReturn,willreturn)
ATTRIBUTE_ENUM(WriteOnly,writeonly)
ATTRIBUTE_ENUM(ZExt,zeroext)
ATTRIBUTE_ENUM(ByRef,byref)
ATTRIBUTE_ENUM(ByVal,byval)
ATTRIBUTE_ENUM(ElementType,elementtype)
ATTRIBUTE_ENUM(InAlloca,inalloca)
ATTRIBUTE_ENUM(Preallocated,preallocated)
ATTRIBUTE_ENUM(StructRet,sret)
ATTRIBUTE_ENUM(Alignment,align)
ATTRIBUTE_ENUM(AllocKind,allockind)
ATTRIBUTE_ENUM(AllocSize,allocsize)
ATTRIBUTE_ENUM(Dereferenceable,dereferenceable)
ATTRIBUTE_ENUM(DereferenceableOrNull,dereferenceable_or_null)
ATTRIBUTE_ENUM(StackAlignment,alignstack)
ATTRIBUTE_ENUM(UWTable,uwtable)
ATTRIBUTE_ENUM(VScaleRange,vscale_range)
#undef ATTRIBUTE_ENUM

#ifndef ATTRIBUTE_STRBOOL
#define ATTRIBUTE_STRBOOL(FIRST, SECOND) ATTRIBUTE_ALL(FIRST, SECOND)
#endif

ATTRIBUTE_STRBOOL(ApproxFuncFPMath,approx-func-fp-math)
ATTRIBUTE_STRBOOL(LessPreciseFPMAD,less-precise-fpmad)
ATTRIBUTE_STRBOOL(NoInfsFPMath,no-infs-fp-math)
ATTRIBUTE_STRBOOL(NoInlineLineTables,no-inline-line-tables)
ATTRIBUTE_STRBOOL(NoJumpTables,no-jump-tables)
ATTRIBUTE_STRBOOL(NoNansFPMath,no-nans-fp-math)
ATTRIBUTE_STRBOOL(NoSignedZerosFPMath,no-signed-zeros-fp-math)
ATTRIBUTE_STRBOOL(ProfileSampleAccurate,profile-sample-accurate)
ATTRIBUTE_STRBOOL(UnsafeFPMath,unsafe-fp-math)
ATTRIBUTE_STRBOOL(UseSampleProfile,use-sample-profile)
#undef ATTRIBUTE_STRBOOL

#undef ATTRIBUTE_ALL
#endif

#ifdef GET_ATTR_ENUM
#undef GET_ATTR_ENUM
FirstEnumAttr = 1,
AllocAlign = 1,
AllocatedPointer = 2,
AlwaysInline = 3,
ArgMemOnly = 4,
Builtin = 5,
Cold = 6,
Convergent = 7,
DisableSanitizerInstrumentation = 8,
Hot = 9,
ImmArg = 10,
InReg = 11,
InaccessibleMemOnly = 12,
InaccessibleMemOrArgMemOnly = 13,
InlineHint = 14,
JumpTable = 15,
MinSize = 16,
MustProgress = 17,
Naked = 18,
Nest = 19,
NoAlias = 20,
NoBuiltin = 21,
NoCallback = 22,
NoCapture = 23,
NoCfCheck = 24,
NoDuplicate = 25,
NoFree = 26,
NoImplicitFloat = 27,
NoInline = 28,
NoMerge = 29,
NoProfile = 30,
NoRecurse = 31,
NoRedZone = 32,
NoReturn = 33,
NoSanitizeBounds = 34,
NoSanitizeCoverage = 35,
NoSync = 36,
NoUndef = 37,
NoUnwind = 38,
NonLazyBind = 39,
NonNull = 40,
NullPointerIsValid = 41,
OptForFuzzing = 42,
OptimizeForSize = 43,
OptimizeNone = 44,
ReadNone = 45,
ReadOnly = 46,
Returned = 47,
ReturnsTwice = 48,
SExt = 49,
SafeStack = 50,
SanitizeAddress = 51,
SanitizeHWAddress = 52,
SanitizeMemTag = 53,
SanitizeMemory = 54,
SanitizeThread = 55,
ShadowCallStack = 56,
Speculatable = 57,
SpeculativeLoadHardening = 58,
StackProtect = 59,
StackProtectReq = 60,
StackProtectStrong = 61,
StrictFP = 62,
SwiftAsync = 63,
SwiftError = 64,
SwiftSelf = 65,
WillReturn = 66,
WriteOnly = 67,
ZExt = 68,
LastEnumAttr = 68,
FirstTypeAttr = 69,
ByRef = 69,
ByVal = 70,
ElementType = 71,
InAlloca = 72,
Preallocated = 73,
StructRet = 74,
LastTypeAttr = 74,
FirstIntAttr = 75,
Alignment = 75,
AllocKind = 76,
AllocSize = 77,
Dereferenceable = 78,
DereferenceableOrNull = 79,
StackAlignment = 80,
UWTable = 81,
VScaleRange = 82,
LastIntAttr = 82,
#endif

#ifdef GET_ATTR_COMPAT_FUNC
#undef GET_ATTR_COMPAT_FUNC
static inline bool hasCompatibleFnAttrs(const Function &Caller,
                                        const Function &Callee) {
  bool Ret = true;

  Ret &= isEqual<SanitizeAddressAttr>(Caller, Callee);
  Ret &= isEqual<SanitizeThreadAttr>(Caller, Callee);
  Ret &= isEqual<SanitizeMemoryAttr>(Caller, Callee);
  Ret &= isEqual<SanitizeHWAddressAttr>(Caller, Callee);
  Ret &= isEqual<SanitizeMemTagAttr>(Caller, Callee);
  Ret &= isEqual<SafeStackAttr>(Caller, Callee);
  Ret &= isEqual<ShadowCallStackAttr>(Caller, Callee);
  Ret &= isEqual<UseSampleProfileAttr>(Caller, Callee);
  Ret &= isEqual<NoProfileAttr>(Caller, Callee);

  return Ret;
}

static inline void mergeFnAttrs(Function &Caller,
                                const Function &Callee) {
  setAND<NoInfsFPMathAttr>(Caller, Callee);
  setAND<NoNansFPMathAttr>(Caller, Callee);
  setAND<ApproxFuncFPMathAttr>(Caller, Callee);
  setAND<NoSignedZerosFPMathAttr>(Caller, Callee);
  setAND<UnsafeFPMathAttr>(Caller, Callee);
  setOR<NoImplicitFloatAttr>(Caller, Callee);
  setOR<NoJumpTablesAttr>(Caller, Callee);
  setOR<ProfileSampleAccurateAttr>(Caller, Callee);
  setOR<SpeculativeLoadHardeningAttr>(Caller, Callee);
  adjustCallerSSPLevel(Caller, Callee);
  adjustCallerStackProbes(Caller, Callee);
  adjustCallerStackProbeSize(Caller, Callee);
  adjustMinLegalVectorWidth(Caller, Callee);
  adjustNullPointerValidAttr(Caller, Callee);
  setAND<MustProgressAttr>(Caller, Callee);
  setAND<LessPreciseFPMADAttr>(Caller, Callee);
}

#endif
#ifdef GET_ATTR_PROP_TABLE
#undef GET_ATTR_PROP_TABLE
static const uint8_t AttrPropTable[] = {
0 | AttributeProperty::ParamAttr,
0 | AttributeProperty::ParamAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::ParamAttr,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::ParamAttr,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::ParamAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr | AttributeProperty::ParamAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr | AttributeProperty::ParamAttr,
0 | AttributeProperty::FnAttr | AttributeProperty::ParamAttr,
0 | AttributeProperty::ParamAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::ParamAttr,
0 | AttributeProperty::ParamAttr,
0 | AttributeProperty::ParamAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr | AttributeProperty::ParamAttr,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr,
0 | AttributeProperty::ParamAttr,
0 | AttributeProperty::ParamAttr,
0 | AttributeProperty::ParamAttr,
0 | AttributeProperty::ParamAttr,
0 | AttributeProperty::FnAttr | AttributeProperty::ParamAttr,
0 | AttributeProperty::ParamAttr,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr,
0 | AttributeProperty::FnAttr | AttributeProperty::ParamAttr,
0 | AttributeProperty::FnAttr,
0 | AttributeProperty::FnAttr,
};
#endif
