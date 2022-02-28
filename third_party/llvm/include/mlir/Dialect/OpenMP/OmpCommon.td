def MemoryOrderKindseq_cst : I32EnumAttrCase<"seq_cst", 0>;
def MemoryOrderKindacq_rel : I32EnumAttrCase<"acq_rel", 1>;
def MemoryOrderKindacquire : I32EnumAttrCase<"acquire", 2>;
def MemoryOrderKindrelease : I32EnumAttrCase<"release", 3>;
def MemoryOrderKindrelaxed : I32EnumAttrCase<"relaxed", 4>;
def MemoryOrderKind: I32EnumAttr<
  "ClauseMemoryOrderKind",
  "MemoryOrderKind Clause",
  [MemoryOrderKindseq_cst,MemoryOrderKindacq_rel,MemoryOrderKindacquire,MemoryOrderKindrelease,MemoryOrderKindrelaxed]> {
    let cppNamespace = "::mlir::omp";
    let genSpecializedAttr = 0;
}
def MemoryOrderKindAttr : EnumAttr<OpenMP_Dialect, MemoryOrderKind, "memoryorderkind">;
def OrderKindconcurrent : I32EnumAttrCase<"concurrent", 1>;
def OrderKind: I32EnumAttr<
  "ClauseOrderKind",
  "OrderKind Clause",
  [OrderKindconcurrent]> {
    let cppNamespace = "::mlir::omp";
    let genSpecializedAttr = 0;
}
def OrderKindAttr : EnumAttr<OpenMP_Dialect, OrderKind, "orderkind">;
def ProcBindKindprimary : I32EnumAttrCase<"primary", 0>;
def ProcBindKindmaster : I32EnumAttrCase<"master", 1>;
def ProcBindKindclose : I32EnumAttrCase<"close", 2>;
def ProcBindKindspread : I32EnumAttrCase<"spread", 3>;
def ProcBindKind: I32EnumAttr<
  "ClauseProcBindKind",
  "ProcBindKind Clause",
  [ProcBindKindprimary,ProcBindKindmaster,ProcBindKindclose,ProcBindKindspread]> {
    let cppNamespace = "::mlir::omp";
    let genSpecializedAttr = 0;
}
def ProcBindKindAttr : EnumAttr<OpenMP_Dialect, ProcBindKind, "procbindkind">;
def ScheduleKindStatic : I32EnumAttrCase<"Static", 0>;
def ScheduleKindDynamic : I32EnumAttrCase<"Dynamic", 1>;
def ScheduleKindGuided : I32EnumAttrCase<"Guided", 2>;
def ScheduleKindAuto : I32EnumAttrCase<"Auto", 3>;
def ScheduleKindRuntime : I32EnumAttrCase<"Runtime", 4>;
def ScheduleKind: I32EnumAttr<
  "ClauseScheduleKind",
  "ScheduleKind Clause",
  [ScheduleKindStatic,ScheduleKindDynamic,ScheduleKindGuided,ScheduleKindAuto,ScheduleKindRuntime]> {
    let cppNamespace = "::mlir::omp";
    let genSpecializedAttr = 0;
}
def ScheduleKindAttr : EnumAttr<OpenMP_Dialect, ScheduleKind, "schedulekind">;
