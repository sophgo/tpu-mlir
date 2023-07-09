# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

# output to specific format

# CSV, RAW Database, Firefox JSON...

# The internal data structure is Structured Object (stay close to python object)

# Serialize use sqlite, (but how to unify the python world)

from dataclasses import dataclass
import dataclasses as dc
from enum import Enum
import json
from typing import List, Hashable


class RefObj:
    def __hash__(self) -> int:
        return id(self)


# --------------------------------------------------------------------------------

# from https://github.com/firefox-devtools/profiler/blob/main/src/types/profile.js


class WeightType(Enum):
    samples = "samples"
    tracing_ms = "tracing-ms"
    bytes = "bytes"


@dataclass(eq=False)
class Lib(RefObj):
    arch: str  # string, // e.g. "x86_64"
    name: str  # string, // e.g. "firefox"
    path: str  # string, // e.g. "/Applications/FirefoxNightly.app/Contents/MacOS/firefox"
    debugName: str  # string, // e.g. "firefox", or "firefox.pdb" on Windows
    debugPath: str  # string, // e.g. "/Applications/FirefoxNightly.app/Contents/MacOS/firefox"
    breakpadId: str  # string, // e.g. "E54D3AF274383256B9F6144F83F3F7510"

    # The codeId is currently always null.
    # In the future, it will have the following values:
    #  - On macOS, it will still be null.
    #  - On Linux / Android, it will have the full GNU build id. (The breakpadId
    #    is also based on the build id, but truncates some information.)
    #    This lets us obtain unstripped system libraries on Linux distributions
    #    which have a "debuginfod" server, and we can use those unstripped binaries
    #    for symbolication.
    #  - On Windows, it will be the codeId for the binary (.exe / .dll), as used
    #    by Windows symbol servers. This will allow us to get assembly code for
    #    Windows system libraries for profiles which were captured on another machine.
    codeId: str  # string | null, // e.g. "6132B96B70fd000"


@dataclass(eq=False)
class Category(RefObj):
    name: str  # string,
    color: str  # string,
    subcategories: list  # string[],


@dataclass
class Page:
    # Tab ID of the page. This ID is the same for all the pages inside a tab's
    # session history.
    # tabID: # TabID,
    # ID of the JS `window` object in a `Document`. It's unique for every page.
    innerWindowID: int  # InnerWindowID,
    #   // Url of this page.
    #   url: # string,
    #   // Each page describes a frame in websites. A frame can either be the top-most
    #   // one or inside of another one. For the children frames, `embedderInnerWindowID`
    #   // points to the innerWindowID of the parent (embedder). It's `0` if there is
    #   // no embedder, which means that it's the top-most frame. That way all pages
    #   // can create a tree of pages that can be navigated.
    embedderInnerWindowID: int  # number,
    #   // If true, this page has been opened in a private browsing window.
    #   // It's optional because it appeared in Firefox 98, and is absent before when
    #   // capturing was disabled when a private browsing window was open.
    #   // The property is always present in Firefox 98+.
    isPrivateBrowsing: bool  # boolean,


@dataclass
class PausedRange:
    #   // null if the profiler was already paused at the beginning of the period of
    #   // time that was present in the profile buffer
    startTime: float  # Milliseconds | null,
    #   // null if the profiler was still paused when the profile was captured
    endTime: float  # Milliseconds | null,
    reason: str  # 'profiler-paused' | 'collecting',


@dataclass
class JsTracerTable:
    events: list  # Array<IndexIntoStringTable>,
    timestamps: list  # Array<Microseconds>,
    durations: list  # Array<Microseconds | null>,
    line: list  # Array<number | null>, // Line number.
    column: list  # Array<number | null>, // Column number.
    length: int  # number,


@dataclass
class CounterSamplesTable:
    time: list  # Milliseconds[],
    # The number of times the Counter's "number" was changed since the previous sample.
    # This property was mandatory until the format version 42, it was made optional in 43.
    number: list  # number[],
    #   // The count of the data, for instance for memory this would be bytes.
    count: list  # number[],
    length: int


@dataclass
class Counter:
    name: str  # string,
    category: str  # string,
    description: str  # string,
    pid: int  # Pid,
    mainThreadIndex: int  # ThreadIndex,
    sampleGroups: dict  # $ReadOnlyArray<{|
    #     id: # number,
    #     samples: # CounterSamplesTable,
    #   |}>,


@dataclass
class ProfilerOverheadStats:
    maxCleaning: float  # Microseconds,
    maxCounter: float  # Microseconds,
    maxInterval: float  # Microseconds,
    maxLockings: float  # Microseconds,
    maxOverhead: float  # Microseconds,
    maxThread: float  # Microseconds,
    meanCleaning: float  # Microseconds,
    meanCounter: float  # Microseconds,
    meanInterval: float  # Microseconds,
    meanLockings: float  # Microseconds,
    meanOverhead: float  # Microseconds,
    meanThread: float  # Microseconds,
    minCleaning: float  # Microseconds,
    minCounter: float  # Microseconds,
    minInterval: float  # Microseconds,
    minLockings: float  # Microseconds,
    minOverhead: float  # Microseconds,
    minThread: float  # Microseconds,
    overheadDurations: float  # Microseconds,
    overheadPercentage: float  # Microseconds,
    profiledDuration: float  # Microseconds,
    samplingCount: float  # Microseconds,


@dataclass
class ProfilerConfiguration:
    threads: list  # string[],
    features: list  # string[],
    capacity: int  # Bytes,
    duration: int  # number,
    #   // Optional because that field is introduced in Firefox 72.
    #   // Active Tab ID indicates a Firefox tab. That field allows us to
    #   // create an "active tab view".
    #   // `0` means null value. Firefox only outputs `0` and not null, that's why we
    #   // should take care of this case while we are consuming it. If it's `0`, we
    #   // should revert back to the full view since there isn't enough data to show
    #   // the active tab view.
    activeTabID: int  # ?TabID,


@dataclass
class ProfilerOverheadSamplesTable:
    counters: list  # Array<Microseconds>,
    expiredMarkerCleaning: list  # Array<Microseconds>,
    locking: list  # Array<Microseconds>,
    threads: list  # Array<Microseconds>,
    time: list  # Array<Milliseconds>,
    length: int  # number,


@dataclass
class ProfilerOverhead:
    samples: ProfilerOverheadSamplesTable  # ProfilerOverheadSamplesTable,
    #   // There is no statistics object if there is no sample.
    statistics: ProfilerOverheadStats  # ProfilerOverheadStats?,
    pid: int  # Pid,
    mainThreadIndex: int  # ThreadIndex,


@dataclass
class ExtensionTable:
    baseURL: list = dc.field(default_factory=list)  # string[],
    id: list = dc.field(default_factory=list)  # string[],
    name: list = dc.field(default_factory=list)  # string[],
    length: int = dc.field(default_factory=list)  # number,


@dataclass
class ProgressGraphData:
    #   // A percentage that describes the visual completeness of the webpage, ranging from 0% - 100%
    percent: float  # number,
    #   // The time in milliseconds which the sample was taken.
    #   // This can be null due to https://github.com/sitespeedio/browsertime/issues/1746.
    timestamp: float  # Milliseconds | null,


@dataclass
class VisualMetrics:
    FirstVisualChange: int  # number,
    LastVisualChange: int  # number,
    SpeedIndex: int  # number,
    VisualProgress: list  # ProgressGraphData[],
    #   // Contentful and Perceptual values may be missing. They're generated only if
    #   // the user specifies the options --visualMetricsContentful and
    #   // --visualMetricsPerceptual in addition to --visualMetrics.
    ContentfulSpeedIndex: int  # number?,
    ContentfulSpeedIndexProgress: list  # ProgressGraphData[],
    PerceptualSpeedIndex: int  # number,
    PerceptualSpeedIndexProgress: list  # ProgressGraphData[],
    #   // VisualReadiness and VisualCompleteXX values are generated in
    #   // https://github.com/sitespeedio/browsertime/blob/main/lib/video/postprocessing/visualmetrics/extraMetrics.js
    VisualReadiness: int  # number,
    VisualComplete85: int  # number,
    VisualComplete95: int  # number,
    VisualComplete99: int  # number,


# // Object that holds the units of samples table values. Some of the values can be
# // different depending on the platform, e.g. threadCPUDelta.
# // See https://searchfox.org/mozilla-central/rev/851bbbd9d9a38c2785a24c13b6412751be8d3253/tools/profiler/core/platform.cpp#2601-2606
@dataclass
class SampleUnits:
    time: str = "\u00b5s"  # 'ms',
    eventDelay: str = "\u00b5s"  # 'ms',
    # ThreadCPUDeltaUnit, 'ns' | 'µs' | 'variable CPU cycles';
    threadCPUDelta: str = "\u00b5s"


@dataclass
class ExtraProfileInfoSection:
    label: str  # string,
    entries: list


@dataclass
class ProfileMeta:
    appBuildID: str  # string?,
    arguments: str  # string?,
    startTime: float  # Milliseconds,
    categories: list  # Array<Category>?,

    CPUName: str  # string?,
    mainMemory: int  # Bytes?,
    sourceURL: str  # string?,
    physicalCPUs: int  # number?,
    logicalCPUs: int  # number,

    markerSchema: list  # MarkerSchema[],

    device: str = "EVB"  # string?,
    misc: str = "commit:abcdefg"  # string,
    stackwalk: int = 0  # 0 | 1,
    interval: float = 0.01  # Milliseconds,
    platform: str = "Linux"
    updateChannel: str = "release"  # ?
    sampleUnits: SampleUnits = SampleUnits()  # SampleUnits?,
    symbolicated: bool = True  # boolean?,
    version: int = 27  # number,
    preprocessedProfileVersion: int = 45  # number,
    product: str = "TPU-MLIR"  # 'Firefox' | string,
    oscpu: str = "TPU Sophgo"  # string?,
    processType: int = 0  # number,


@dataclass
class SerializableThread:
    #   ...$Diff<Thread, { stringTable: # UniqueStringArray }>,
    stringArray: str  # string[],


@dataclass
class SerializableProfile:
    #   ...Profile,
    threads: list  # SerializableThread[],


# --------------------------------------------------------------------------------


class Table:
    def __init__(self, type=None) -> None:
        self.table = {}
        self.unhashable = []
        self.type = type

    def get_id(self, obj):
        if obj is None:
            return None

        if self.type:
            assert self.type == type(obj)
        else:
            self.type = type(obj)

        if isinstance(obj, Hashable):
            assert self.unhashable == []
            if obj not in self.table:
                self.table[obj] = len(self.table)
            return self.table[obj]
        else:
            assert self.table == {}
            try:
                return self.unhashable.index(obj)
            except ValueError:
                self.unhashable.append(obj)
                return len(obj) - 1

    def __getitem__(self, key):
        return self.get_id(key)

    def records(self):
        if self.table:
            return list(self.table.keys())
        return self.unhashable

    def __repr__(self) -> str:
        return str(self.records())


class TableEncoder(json.JSONEncoder):
    def tablToJson(self, o):
        rd = o.records()
        if len(rd) > 0:
            if isinstance(rd[0], (str, Lib)):
                return rd
            try:
                name = (x.name for x in dc.fields(rd[0]) if x.repr == True)
                out = {n: [getattr(r, n) for r in rd] for n in name}

                # edge case
                # The Firefox needs only one weightType
                if "weightType" in out and isinstance(out["weightType"][0], WeightType):
                    out["weightType"] = out["weightType"][0]

                out["length"] = len(rd)
                return out
            except:
                raise TypeError(f"Table: {o}")
        elif o.type:
            name = (x.name for x in dc.fields(o.type) if x.repr == True)
            return {n: [] for n in name}
        return []

    def default(self, o):
        try:
            return super().default(o)
        except TypeError:
            if isinstance(o, Table):
                return self.tablToJson(o)
            elif isinstance(o, Enum):
                return o.value
            raise TypeError(f"type:{type(o)}. {o}")


@dataclass(eq=False)
class Resource(RefObj):
    str_table: Table = dc.field(repr=False, hash=None, compare=False)
    lib_table: Table = dc.field(repr=False, hash=None, compare=False)
    name: str
    host: str
    type: int = 0
    lib: int = None

    def __post_init__(self):
        self.name = self.str_table.get_id(self.name)
        self.host = self.str_table.get_id(self.host)
        if self.lib:
            self.lib = self.lib_table.get_id(self.lib)


@dataclass(eq=False)
class Function(RefObj):
    str_table: Table = dc.field(repr=False, hash=None, compare=False)
    resource_table: Table = dc.field(repr=False, hash=None, compare=False)
    name: str
    fileName: str = None
    lineNumber: int = None
    columnNumber: int = None
    isJS: bool = False
    # The resource describes "Which bag of code did this function come from?".
    # For JS functions, the resource is of type addon, webhost, otherhost, or url.
    # For native functions, the resource is of type library.
    # For labels and for other unidentified functions, we set the resource to -1.
    resource: int = -1
    relevantForJS: bool = False

    def __post_init__(self):
        self.name = self.str_table.get_id(self.name)
        if self.fileName is not None:
            self.fileName = self.str_table.get_id(self.fileName)
        if self.resource > -1:
            self.resource = self.resource_table.get_id(self.resource)


@dataclass(eq=False)
class nativeSymbol(RefObj):
    str_table: Table = dc.field(repr=False, hash=None, compare=False)
    lib_table: Table = dc.field(repr=False, hash=None, compare=False)
    name: str
    libIndex: int
    address: int
    functionSize: int = None

    def __post_init__(self):
        self.name = self.str_table.get_id(self.name)
        self.libIndex = self.lib_table.get_id(self.libIndex)


@dataclass(eq=False)
class Frame(RefObj):
    str_table: Table = dc.field(repr=False, hash=None, compare=False)
    func_table: Table = dc.field(repr=False, hash=None, compare=False)
    symbol_table: Table = dc.field(repr=False, hash=None, compare=False)
    func: Function
    # The inline depth for this frame. If there is an inline stack at an address,
    # we create multiple frames with the same address, one for each depth.
    # The outermost frame always has depth 0.
    #
    # Example:
    # If the raw stack is 0x10 -> 0x20 -> 0x30, and symbolication adds two inline frames
    # for 0x10, no inline frame for 0x20, and one inline frame for 0x30, then the
    # symbolicated stack will be the following:
    #
    # func: #        outer1 -> inline1a -> inline1b -> outer2 -> outer3 -> inline3a
    # address: #     0x10   -> 0x10     -> 0x10     -> 0x20   -> 0x30   -> 0x30
    # inlineDepth: #    0   ->    1     ->    2     ->    0   ->    0   ->    1
    #
    # Background:
    # When a compiler performs an inlining optimization, it removes a call to a function
    # and instead generates the code for the called function directly into the outer
    # function. But it remembers which instructions were the result of this inlining,
    # so that information about the inlined function can be recovered from the debug
    # information during symbolication, based on the instruction address.
    # The compiler can choose to do inlining multiple levels deep: # An instruction can
    # be the result of a whole "inline stack" of functions.
    # Before symbolication, all frames have depth 0. During symbolication, we resolve
    # addresses to inline stacks, and create extra frames with non-zero depths as needed.
    #
    # The frames of an inline stack at an address all have the same address and the same
    # nativeSymbol, but each has a different func and line.
    category: int = None
    subcategory: int = None
    inlineDepth: int = 0
    # The library which this address is relative to is given by the frame's nativeSymbol:
    # frame -> nativeSymbol -> lib.
    address: int = -1
    # The symbol index (referring into this thread's nativeSymbols table) corresponding
    # to symbol that covers the frame address of this frame. Only non-null for native
    # frames (e.g. C / C++ / Rust code). Null before symbolication.
    nativeSymbol: nativeSymbol = None
    innerWindowID: int = None
    implementation: str = None
    line: int = None
    column: int = None

    def __post_init__(self):
        self.func = self.func_table.get_id(self.func)
        self.implementation = self.str_table.get_id(self.implementation)
        self.nativeSymbol = self.symbol_table.get_id(self.nativeSymbol)


class MarkerPhase(Enum):
    Instant = 0
    Interval = 1
    IntervalStart = 2
    IntervalEnd = 3


class markerSchema:
    name: str  # e.g. "CC"
    tooltipLabel: str  # ? e.g. "Cycle Collect"
    tableLabel: str  # ? e.g. "{marker.data.eventType} – DOMEvent"
    chartLabel: str  # ?
    display: List[str] = dc.field(default_factory=list)  #  // The locations to display
    data: List[dict] = dc.field(
        default_factory=list
    )  # dict {key, label, format, searchable} | {label, value}


@dataclass(eq=False)
class Marker(RefObj):
    str_table: Table = dc.field(
        default_factory=list, repr=False, hash=None, compare=False
    )
    name: str = None
    startTime: float = None
    endTime: float = None
    phase: MarkerPhase = MarkerPhase.Instant
    category: int = 0
    data: dict = dc.field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.name, str):
            self.name = self.str_table.get_id(self.name)


@dataclass(eq=False)
class Stack(RefObj):
    frame_table: Table = dc.field(repr=False, hash=None, compare=False)
    stack_table: Table = dc.field(repr=False, hash=None, compare=False)
    frame: Frame
    category: int
    subcategory: int = None
    prefix: int = None

    def __post_init__(self):
        self.frame = self.frame_table.get_id(self.frame)
        if isinstance(self.prefix, Stack):
            self.prefix = self.stack_table.get_id(self.prefix)


@dataclass(eq=False)
class Sample(RefObj):
    stack_table: Table = dc.field(repr=False, hash=None, compare=False)
    stack: Stack
    threadCPUDelta: int
    time: float
    weight: None = None
    weightType: WeightType = WeightType.tracing_ms
    # threadId: Thread

    def __post_init__(self):
        self.stack = self.stack_table.get_id(self.stack)


class MarkerPayload(markerSchema):
    pass


@dataclass
class RawMarker:
    data: MarkerPayload
    name: str
    startTime: float
    endTime: float
    phase: MarkerPhase
    category: Category
    threadId: int


@dataclass(eq=False)
class Thread(RefObj):
    # Top -> Bottom
    # // static information
    # stack{
    #   frame{
    #      function
    #    }
    # }
    #
    # dynamic information
    # sample{
    #  timing
    #  static information
    # }
    lib_list: Table = dc.field(repr=False, hash=None, compare=False)
    name: str  # string,
    processType: str
    processStartupTime: float  # Milliseconds,
    registerTime: float  # Milliseconds,
    isMainThread: bool  # boolean,
    processName: str  # string?,
    pid: int  # Pid,
    tid: int  # Tid,
    markers: Table = dc.field(default_factory=Table)
    samples: Table = dc.field(default_factory=Table)
    stackTable: Table = dc.field(default_factory=Table)
    frameTable: Table = dc.field(default_factory=Table)
    funcTable: Table = dc.field(default_factory=Table)
    resourceTable: Table = dc.field(default_factory=Table)
    nativeSymbols: Table = dc.field(default_factory=Table)
    stringArray: Table = dc.field(default_factory=Table)
    pausedRanges: list = dc.field(default_factory=list)
    processShutdownTime: float = None
    unregisterTime: float = None

    def __post_init__(self):
        self.markers.type = Marker
        self.stringArray.type = str

    def add_resource(self, *args, **kargs):
        func = Resource(self.stringArray, self.lib_list, *args, **kargs)
        self.resourceTable.get_id(func)
        return func

    def add_function(self, *args, **kargs):
        func = Function(self.stringArray, self.resourceTable, *args, **kargs)
        self.funcTable.get_id(func)
        return func

    def add_symbol(self, *args, **kargs):
        symbol = nativeSymbol(self.stringArray, self.lib_list, *args, **kargs)
        self.nativeSymbols.get_id(symbol)
        return symbol

    def add_frame(self, *args, **kargs):
        frame = Frame(
            self.stringArray, self.funcTable, self.nativeSymbols, *args, **kargs
        )
        self.frameTable.get_id(frame)
        return frame

    def add_stack(self, *args, **kargs):
        stack = Stack(self.frameTable, self.stackTable, *args, **kargs)
        self.stackTable.get_id(stack)
        return stack

    def add_sample(self, *args, **kargs):
        sample = Sample(self.stackTable, *args, **kargs)
        self.samples.get_id(sample)
        return sample

    def add_marker(self, *args, **kargs):
        marker = Marker(self.stringArray, *args, **kargs)
        self.markers.get_id(marker)
        return marker


@dataclass(eq=False)
class Profile(RefObj):
    meta: ProfileMeta  # ProfileMeta,
    threads: list = dc.field(default_factory=list)
    libs: Table = dc.field(default_factory=Table)
    counters: list = dc.field(default_factory=list)
    pages: list = dc.field(default_factory=list)

    def add_lib(self, *args, **kargs):
        lib = Lib(*args, **kargs)
        self.libs.get_id(lib)
        return lib

    def add_thread(self, *args, **kargs):
        thread = Thread(self.libs, *args, **kargs)
        self.threads.append(thread)
        return thread


# --------------------------------------------------------------------------------
# test
"""
categories = [
    # blue, green, lightblue, orange, yellow, purple, grey, transparent
    Category("Idle", "transparent", ["Other"]),
    Category("Other", "grey", ["Other"]),
    Category("TIU", "orange", ["TPU", "Compute", "Local Memory"]),
    Category("DMA", "blue", ["TPU", "Data"]),
]
meta = ProfileMeta(
    appBuildID="resnet50",
    arguments="<1x2x3x4xf32>",
    startTime=0,
    categories=categories,
    CPUName="bm1684x",
    mainMemory=2**32,
    sourceURL="sophgo.com",
    physicalCPUs=1,
    logicalCPUs=2,
    markerSchema=[],
)

profile = Profile(
    meta=meta,
    libs=[],
    pages=[],
    counters=[],
)
thread1 = Thread(profile.libs, "main", "TIU", 0, 1, True, "bm1684x", 2, 3)
profile.threads.append(thread1)

rs = thread1.add_resource("decoder", "vpp")
func = thread1.add_function("func_1(tensor_a)", "file_A", 120)
func2 = thread1.add_function("func_2(tensor_b)", "file_B", 110)
frame = thread1.add_frame(func, 2, 0)
frame2 = thread1.add_frame(func2, 3, 0)
stack1 = thread1.add_stack(frame, 2, 0)
stack2 = thread1.add_stack(frame2, 3, 0, stack1)
stack3 = thread1.add_stack(frame2, 3, 0, stack2)
thread1.add_sample(stack1, 100.0, 1000, 0.03)
thread1.add_sample(stack2, 200.0, 1000, 20)
thread1.add_sample(stack3, 900.0, 1000, 20)
thread1.add_sample(stack1, 1200.0, 100, 50)


with open("out.json", "w") as f:
    json.dump(dc.asdict(profile), f, indent=2, cls=TableEncoder)
"""
