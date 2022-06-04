def CancellationConstructTypeparallel : I32EnumAttrCase<"Parallel", 0, "parallel">;
def CancellationConstructTypeloop : I32EnumAttrCase<"Loop", 1, "loop">;
def CancellationConstructTypesections : I32EnumAttrCase<"Sections", 2, "sections">;
def CancellationConstructTypetaskgroup : I32EnumAttrCase<"Taskgroup", 3, "taskgroup">;
def CancellationConstructType: I32EnumAttr<
  "ClauseCancellationConstructType",
  "CancellationConstructType Clause",
  [CancellationConstructTypeparallel,CancellationConstructTypeloop,CancellationConstructTypesections,CancellationConstructTypetaskgroup]> {
    let cppNamespace = "::mlir::omp";
    let genSpecializedAttr = 0;
}
def CancellationConstructTypeAttr : EnumAttr<OpenMP_Dialect, CancellationConstructType, "cancellationconstructtype">;
def MemoryOrderKindseq_cst : I32EnumAttrCase<"Seq_cst", 0, "seq_cst">;
def MemoryOrderKindacq_rel : I32EnumAttrCase<"Acq_rel", 1, "acq_rel">;
def MemoryOrderKindacquire : I32EnumAttrCase<"Acquire", 2, "acquire">;
def MemoryOrderKindrelease : I32EnumAttrCase<"Release", 3, "release">;
def MemoryOrderKindrelaxed : I32EnumAttrCase<"Relaxed", 4, "relaxed">;
def MemoryOrderKind: I32EnumAttr<
  "ClauseMemoryOrderKind",
  "MemoryOrderKind Clause",
  [MemoryOrderKindseq_cst,MemoryOrderKindacq_rel,MemoryOrderKindacquire,MemoryOrderKindrelease,MemoryOrderKindrelaxed]> {
    let cppNamespace = "::mlir::omp";
    let genSpecializedAttr = 0;
}
def MemoryOrderKindAttr : EnumAttr<OpenMP_Dialect, MemoryOrderKind, "memoryorderkind">;
def OrderKindconcurrent : I32EnumAttrCase<"Concurrent", 1, "concurrent">;
def OrderKind: I32EnumAttr<
  "ClauseOrderKind",
  "OrderKind Clause",
  [OrderKindconcurrent]> {
    let cppNamespace = "::mlir::omp";
    let genSpecializedAttr = 0;
}
def OrderKindAttr : EnumAttr<OpenMP_Dialect, OrderKind, "orderkind">;
def ProcBindKindprimary : I32EnumAttrCase<"Primary", 0, "primary">;
def ProcBindKindmaster : I32EnumAttrCase<"Master", 1, "master">;
def ProcBindKindclose : I32EnumAttrCase<"Close", 2, "close">;
def ProcBindKindspread : I32EnumAttrCase<"Spread", 3, "spread">;
def ProcBindKind: I32EnumAttr<
  "ClauseProcBindKind",
  "ProcBindKind Clause",
  [ProcBindKindprimary,ProcBindKindmaster,ProcBindKindclose,ProcBindKindspread]> {
    let cppNamespace = "::mlir::omp";
    let genSpecializedAttr = 0;
}
def ProcBindKindAttr : EnumAttr<OpenMP_Dialect, ProcBindKind, "procbindkind">;
def ScheduleKindstatic : I32EnumAttrCase<"Static", 0, "static">;
def ScheduleKinddynamic : I32EnumAttrCase<"Dynamic", 1, "dynamic">;
def ScheduleKindguided : I32EnumAttrCase<"Guided", 2, "guided">;
def ScheduleKindauto : I32EnumAttrCase<"Auto", 3, "auto">;
def ScheduleKindruntime : I32EnumAttrCase<"Runtime", 4, "runtime">;
def ScheduleKind: I32EnumAttr<
  "ClauseScheduleKind",
  "ScheduleKind Clause",
  [ScheduleKindstatic,ScheduleKinddynamic,ScheduleKindguided,ScheduleKindauto,ScheduleKindruntime]> {
    let cppNamespace = "::mlir::omp";
    let genSpecializedAttr = 0;
}
def ScheduleKindAttr : EnumAttr<OpenMP_Dialect, ScheduleKind, "schedulekind">;
