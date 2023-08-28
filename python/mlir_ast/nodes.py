#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from typing import List, Union

from collections import OrderedDict, Counter
import textwrap
import re
import mlir.ir
from mlir.dialects import quant

SPACE = "  "


def shape_int(x):
    if x == "*":
        return x

    return int(x)


def parse_outputs_str(outputs_str: str):
    type_end_idx = outputs_str.find(" loc")
    type_str = outputs_str[:type_end_idx]
    loc_start_idx = type_end_idx + 1
    loc_str = outputs_str[loc_start_idx:]

    loc_label = LocLabel.parse(loc_str)
    if type_str.strip() == "()":
        return ([], loc_label)

    if type_str.startswith("("):
        return (Type.parse_inputs_tuple(type_str), loc_label)

    if type_str.count("tensor") + type_str.count("none") > 1:
        return (Type.parse_inputs_tuple(type_str), loc_label)

    return ([Type.parse(type_str)], loc_label)


def with_assert(cls):
    # return cls
    parse_fn = getattr(cls, "parse", None)

    def assert_parse(name: str):
        name = name.strip()
        try:
            res = parse_fn(name)
            dump_str = res.dump()
            assert dump_str == name
        except:
            print(cls.__name__)
            print("orig", name)
            print("dump", dump_str)

            raise ValueError()
        return res

    if parse_fn:
        setattr(cls, "parse", assert_parse)
    return cls


class Node:
    _context = None

    def __init__(self) -> None:
        from .mlir_ast import AST_CONTEXT

        if AST_CONTEXT is not None:
            self._context = AST_CONTEXT

    @property
    def context(self):
        if self._context is None:
            raise ValueError(
                "missing mlir module context, should be created nodes by MlirASTParser"
            )
        return self._context


@with_assert
class OperationType(Node):
    def __init__(self, op_type_name: str, opds: List[str], subcall=False) -> None:
        super().__init__()
        self.op_type_name = op_type_name.strip()
        self.opds = opds
        self.subcall = subcall

    def isa(self, *args):
        return any([i == self.op_type_name for i in args])

    @property
    def unique_opds(self):
        return list(Counter(self.opds).keys())

    @staticmethod
    def parse(operation_name: str):
        """
        "tpu.Conv2D"(%600, %598, %599)
        "tpu.Store"(%602)
        "top.Weight"()
        x call @subfunc_0(%0)
        """
        opds_start_idx = operation_name.find("(") + 1
        opds_str = operation_name.strip()[opds_start_idx:-1]
        opds = list(map(lambda x: x.strip(), opds_str.split(",")))
        opds = [i for i in opds if i != ""]
        subcall = False
        op_type_end_idx = opds_start_idx - 1
        if operation_name.startswith("call"):
            subcall = True
            call_func_start_idx = operation_name.find("@") + 1
            op_type = operation_name[call_func_start_idx:op_type_end_idx]
        else:
            op_type = operation_name[:op_type_end_idx].strip('"')

        return OperationType(op_type, opds, subcall)

    def dump(self):
        opds_str = ", ".join(self.opds)
        if self.subcall:
            return f"call @{self.op_type_name}({opds_str})"
        return f'"{self.op_type_name}"({opds_str})'


class NoneType(Node):
    @property
    def ir(self):
        return mlir.ir.Type.parse("none", self.context.ctx)

    def dump(self):
        return "none"


@with_assert
class Type(Node):
    match_tensor = re.compile("tensor<|none")
    match_tensor_from_line = re.compile("tensor<.*>|none")

    def __init__(self, shape, dtype, address=None) -> None:
        super().__init__()
        self.shape = shape
        self.dtype = dtype
        self.address = address

    @property
    def ir(self):
        return mlir.ir.Type.parse(self.dump(), self.context.ctx)

    def create_a_f32(self):
        return Type(self.shape, "f32", self.address)

    @staticmethod
    def match_tensor_content(line: str) -> Union["Type", List["Type"]]:
        def match_tensor(tensor_start_idx):
            _start_idx = line.find("<", tensor_start_idx)
            tensor_end_idx = line.find(">", _start_idx)
            while line.find("<", _start_idx + 1) >= 0:
                _start_idx = line.find("<", _start_idx + 1)
                tensor_end_idx = line.find(">", tensor_end_idx + 1)
            return tensor_end_idx + 1

        def match_none():
            pass

        multiple = False
        count = 0
        match_res = Type.match_tensor.search(line)
        types = []
        while match_res:
            start_idx = match_res.start()
            if count == 0 and line[start_idx - 1] == "(":
                multiple = True

            if line[start_idx] == "t":
                tensor_end_idx = match_tensor(start_idx)
                types.append(line[start_idx:tensor_end_idx])
                start_idx = tensor_end_idx
            else:
                types.append(line[start_idx : start_idx + 4])
                start_idx += 4

            match_res = Type.match_tensor.search(line)

        if multiple:
            return types
        else:
            return types[0]

    @staticmethod
    def _raw_parse(tensor_str: str):
        """
        1x256x14x14xf32
        1x256x14x14x!quant.uniform<i8:f32, 0.064606035433070863>
        1x25088x!quant.uniform<i8:f32, 0.051140184251968507>, 4362649600 : i64
        1x16x512x!quant.calibrated<f32<-0.92979659999999997:0.92979659999999997>>, 4459302912 : i64
        """

        address = None

        if tensor_str.endswith("i64"):
            address_start_idx = tensor_str.rfind(",")
            tensor_str, address = (
                tensor_str[:address_start_idx],
                tensor_str[address_start_idx + 2 :],
            )

        quant_start_idx = tensor_str.find("!")
        if quant_start_idx >= 0:
            shape_str, dtype = (
                tensor_str[:quant_start_idx],
                tensor_str[quant_start_idx:],
            )
            shape_str = shape_str.rstrip("x")
        else:
            shape_str, dtype = tensor_str.rsplit("x", maxsplit=1)

        shape = list(map(shape_int, shape_str.split("x")))
        dtype = dtype.strip()
        if address is not None:
            address = address.strip()

        return Type(shape, dtype, address)

    @staticmethod
    def dump_type_list(types: List["Type"], force_list=False) -> str:
        if len(types) == 0:
            return "()"
        if len(types) == 1 and not force_list:
            return types[0].dump()
        inner = ", ".join([i.dump() for i in types])
        return f"({inner})"

    @staticmethod
    def parse(dtype_str: str) -> "Type":
        """
        tensor<1x3x384x288xf32>
        tensor<1x3x384x288xf32, 4410114048 : i64>
        none
        """
        if dtype_str.strip() == "none":
            return NoneType()
        return Type._raw_parse(dtype_str.replace("tensor<", "")[:-1])

    @staticmethod
    def parse_output(dtype_str: str) -> List["Type"]:
        if dtype_str.strip() == "none":
            return NoneType()

        if dtype_str.startswith("("):
            return Type.parse_inputs_tuple(dtype_str)

        return [Type.parse(dtype_str)]

    @staticmethod
    def parse_inputs_tuple(type_str: str) -> List["Type"]:
        """
        - ()
        - (tensor<1x3x384x288xf32, 4410114048 : i64>)
        - (tensor<1x3x384x288xf32, 4410114048 : i64>, none)
        - (tensor<1x3x384x288xf32>, tensor<1x64x3x9xf32>, tensor<1x64x1x1xf32>)

        pattern: tensor<[^>]+>
        """
        res = []
        match_tensor_start = Type.match_tensor.search(type_str)
        while match_tensor_start:
            tensor_start_idx = match_tensor_start.start()
            match_tensor_end = Type.match_tensor.search(
                type_str, pos=tensor_start_idx + 1
            )
            if match_tensor_end:
                tensor_end_idx = match_tensor_end.start() - 2
            else:
                tensor_end_idx = None
            tensor_str = type_str[tensor_start_idx:tensor_end_idx]
            if (tensor_str.strip()) == "none":
                res.append(NoneType())
            else:
                res.append(Type.parse(tensor_str.rstrip(" )")))
            match_tensor_start = match_tensor_end

        return res

    def dump(self):
        shape_str = "x".join(map(str, self.shape))
        if self.address is not None:
            return f"tensor<{shape_str}x{self.dtype}, {self.address}>"
        else:
            return f"tensor<{shape_str}x{self.dtype}>"


@with_assert
class LocLabel(Node):
    """
    #loc
    #loc<id>
    """

    def __init__(self, loc_id_str: str) -> None:
        super().__init__()
        self.loc_id_str = loc_id_str.strip()

    @property
    def id(self):
        if len(self.loc_id_str) == 4:
            return -1
        return int(self.loc_id_str[4:])

    def dump(self):
        return f"loc({self.loc_id_str})"

    def parse(loc_label_str):
        """
        loc(#loc<id>)
        """
        return LocLabel(loc_label_str.strip()[4:-1])


@with_assert
class Location(Node):
    def __init__(self, loc_id_str: str, loc_name: str, fused: List[str] = None) -> None:
        super().__init__()
        self.loc_id_str = loc_id_str
        self._loc_name = loc_name
        self.fused = fused

    def to_label(self) -> LocLabel:
        return LocLabel(self.loc_id_str)

    @property
    def isfused(self):
        return self.fused is not None

    @staticmethod
    def parse(location_token: str):
        """
        <>
        #loc = loc(unknown)
        #loc2 = loc("onnx::Conv_2949")
        """
        loc_id_str, _, name_pos = location_token.split(" ", maxsplit=2)

        if "unknown" in name_pos:
            loc_name = "unknown"
        elif "fused" in name_pos:
            fused_id = name_pos[10:-2].split(", ")
            return Location(
                loc_id_str=loc_id_str, loc_name=f"fused_{name_pos}", fused=fused_id
            )
        else:
            loc_name = name_pos[4:-1].strip('"')

        return Location(loc_id_str=loc_id_str, loc_name=loc_name)

    def dump(self):
        loc_name = (
            f'"{self._loc_name}"' if self._loc_name != "unknown" else self._loc_name
        )
        if self.fused is not None:
            fused_name = ", ".join([f"{i}" for i in self.fused])
            return f"{self.loc_id_str} = loc(fused[{fused_name}])"
        else:
            return f"{self.loc_id_str} = loc({loc_name})"


@with_assert
class Attributes(Node):
    match_eq = re.compile("#([^<]+)<([^<]+)>")
    match_m = re.compile(r"([\-.+\w]+ : \w+)")
    match_array = re.compile("(array<[^>+]+>)")
    match_key = re.compile("([\w.]+) =")

    def __init__(self, attributes: OrderedDict) -> None:
        super().__init__()
        self.attributes = attributes

    def __getitem__(self, key):
        return self.attributes.get(key)

    def to_dict(self):
        return self.attributes

    @staticmethod
    def parse(attr_str: str) -> "Attributes":
        """attr_str: {...}"""
        temp = Node()
        attribute = mlir.ir.Attribute.parse(attr_str, temp.context.ctx)
        arr_map = OrderedDict()
        for i in range(len(attribute)):
            attr = attribute[i]
            k, v = str(attr.name), str(attr.attr)
            arr_map[k] = v

        return Attributes(arr_map)

    def dump(self):
        def dump_dict(dic: dict):
            inner_str = []
            for k, v in dic.items():
                if isinstance(v, bool):
                    v = str(v).lower()
                elif isinstance(v, str) and len(v.split()) == 1:
                    v = f"{v}"
                elif isinstance(v, list):
                    v = ", ".join(v)
                    v = f"[{v}]"
                elif isinstance(v, OrderedDict):
                    v = dump_dict(v)
                    v = f"{{{v}}}"

                inner_str.append(f"{k} = {v}")
            inner_str = ", ".join(inner_str)
            return inner_str

        inner_str = dump_dict(self.attributes)

        return f"{{{inner_str}}}"


@with_assert
class Operation(Node):
    # input: List[Operation]
    # dtype/result : List[Dtype]

    # attr: List[Attribute]
    # loc: Location

    # lno: int
    def __init__(
        self,
        opd_ids: List[str],
        op_type: OperationType,
        input_types: List[Type],
        output_types: List[Type],
        loc_label: LocLabel,
        attrs: Attributes = None,
        const: str = None,  # tosa.const
    ) -> None:
        super().__init__()
        # <opd_ids> = <type_name> (<attr>) : <inputs> -> <outputs>
        self.opd_ids = opd_ids
        self.op_type = op_type
        self.input_types = input_types
        self.output_types = output_types
        self._attrs = attrs
        self.loc_label = loc_label
        self.const = const
        self._name = None
        self._parent: GroupOp = None
        self.erased = False

    @property
    def ns_opd_ids(self):
        ns = self.ns
        return [ns + i for i in self.opd_ids]

    @property
    def ns(self) -> str:
        if self._parent is None:
            return ""
        return self._parent.name + "-"

    def erase(self):
        self.erased = True

    @property
    def parent(self) -> "GroupOp":
        return self._parent

    @property
    def name(self):
        if self._name is None:
            return self.context.locid2opname[self.loc_label.loc_id_str]
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def type(self) -> str:
        """top.Input"""
        return self.op_type.op_type_name

    @property
    def attrs(self):
        if self._attrs is None:
            res = {}
        else:
            res = self._attrs.to_dict()
        if len(self.opd_ids) != 1:
            return res

        # align mlir_parser.py
        if not isinstance(self.output_types[0], NoneType):
            element_type = self.output_types[0].ir.element_type
            if quant.UniformQuantizedType.isinstance(element_type):
                quant_type = quant.UniformQuantizedType(element_type)
                res["quant_scale"] = str(quant_type.scale)
                res["quant_zero_point"] = str(quant_type.zero_point)
            if quant.CalibratedQuantizedType.isinstance(element_type):
                quant_type = quant.CalibratedQuantizedType(element_type)
                res["calibrate_min"] = str(quant_type.min)
                res["calibrate_max"] = str(quant_type.max)
        return res

    @property
    def opds(self) -> List[str]:
        return [
            self.context.get_op_name_by_op_id(i) for i in self.op_type.opds if i != "%0"
        ]

    @property
    def outputs(self) -> List[str]:
        return [self.context.get_op_name_by_op_id(i) for i in self.opd_ids]

    @staticmethod
    def parse(operation_line: str) -> "Operation":
        """
        %10:2 = "top.Split"(%2) {axis = 1 : si32, num = 2 : i64} : (tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>) loc(#loc592)
        """
        op_id_str, op_define = operation_line.strip().split("=", maxsplit=1)
        if op_id_str.find(":") > 0:
            opd_ids = []
            op_id_prefix, count = op_id_str.split(":")
            for i in range(int(count)):
                opd_ids.append(f"{op_id_prefix}#{i}")
        else:
            opd_ids = op_id_str.strip().split(", ")

        const = None
        if "tosa" in op_define:
            count_start_idx = op_define.find("<{")
            if count_start_idx != -1:
                count_end_idx = op_define.find("}>") + 2
                const = op_define[count_start_idx:count_end_idx]
                op_define = op_define[:count_start_idx] + op_define[count_end_idx:]

        attr_str = ""
        attr_start_idx = op_define.find("{")
        attr_end_idx = -1

        attr = None
        if attr_start_idx != -1:
            attr_end_idx = op_define.find("}", attr_start_idx) + 1
            attr_str = op_define[attr_start_idx:attr_end_idx]
            attr_str += "}" * (attr_str.count("{") - 1)
            attr = Attributes.parse(attr_str)

        if attr_end_idx != -1:
            inputs_start_idx = attr_end_idx + 2
            type_end_idx = attr_start_idx - 1
        else:
            inputs_start_idx = op_define.find(": ") + 1
            type_end_idx = inputs_start_idx - 1

        inputs_end_idx = op_define.find(" ->", inputs_start_idx)
        inputs_str = op_define[inputs_start_idx:inputs_end_idx]
        op_type_str = op_define[:type_end_idx].strip()

        outputs_start_idx = inputs_end_idx + 3
        outputs_str = op_define[outputs_start_idx:]

        input_types = Type.parse_inputs_tuple(inputs_str)
        output_types, loc_label = parse_outputs_str(outputs_str)

        op_type = OperationType.parse(op_type_str)
        if op_define.strip().startswith("call"):
            return CallFunc(opd_ids, op_type, input_types, output_types, loc_label)

        return Operation(
            opd_ids, op_type, input_types, output_types, loc_label, attr, const
        )

    def dump(self):
        input_types_str = Type.dump_type_list(self.input_types, force_list=True)
        output_types_str = Type.dump_type_list(self.output_types)
        loc_label_str = self.loc_label.dump()

        if "#" in self.opd_ids[0]:
            number = len(self.opd_ids)
            prefix = self.opd_ids[0].split("#")[0]
            opd_str = f"{prefix}:{number}"
        else:
            opd_str = ", ".join(self.opd_ids)
        const_str = ""
        if self.const is not None:
            const_str = f"{const_str} "

        if self._attrs is None:
            return f"{opd_str} = {self.op_type.dump()} {const_str}: {input_types_str} -> {output_types_str} {loc_label_str}"
        else:
            attrs_str = self._attrs.dump()
            return f"{opd_str} = {self.op_type.dump()} {attrs_str} : {input_types_str} -> {output_types_str} {loc_label_str}"


@with_assert
class Return(Operation):
    def __init__(
        self, output_ids: List[str], output_types: List[Type], loc_label: LocLabel
    ) -> None:
        super().__init__(
            [],
            OperationType("func.return", output_ids),
            output_types,  # NOTE: Return op have no input_types in defination
            output_types,
            loc_label,
        )

    @property
    def output_ids(self):
        return self.op_type.opds

    @staticmethod
    def parse(return_line) -> "Return":
        """
        return %1 : tensor<1x133x96x72xf32, 4418519040 : i64> loc(#loc)
        """
        left_str, outputs_str = return_line.strip().split(":", maxsplit=1)
        op_id_start_id = left_str.find("%")
        output_types, loc_label = parse_outputs_str(outputs_str)
        opd_ids = left_str[op_id_start_id:].strip().split(", ")
        return Return(opd_ids, output_types, loc_label)

    def dump(self):
        output_id_str = ", ".join(self.output_ids)
        output_type_str = Type.dump_type_list(self.output_types).strip("()")
        return f"return {output_id_str} : {output_type_str} {self.loc_label.dump()}"


@with_assert
class Yield(Operation):
    def __init__(
        self,
        op_type: OperationType,
        input_types: List[Type],
        output_types: List[Type],
        loc_label: LocLabel,
    ) -> None:
        super().__init__([], op_type, input_types, output_types, loc_label)

    @staticmethod
    def parse(yield_line):
        """
        "tpu.Yield"(%603, %605) : (tensor<1x256x96x72xf32, 4411441152 : i64>, tensor<1x64x96x72xf32, 4427366400 : i64>) -> () loc(#loc1668)
        "tpu.Yield"(%1612) : (tensor<1x133x96x72xf32, 4418519040 : i64>) -> () loc(#loc646)
        """
        op_type_str, op_define = yield_line.strip().split(":", maxsplit=1)
        op_type = OperationType.parse(op_type_str)
        inputs_end_idx = op_define.find(" ->")
        inputs_str = op_define[:inputs_end_idx]

        outputs_start_idx = inputs_end_idx + 3
        outputs_str = op_define[outputs_start_idx:]

        input_types = Type.parse_inputs_tuple(inputs_str)
        output_types, loc_label = parse_outputs_str(outputs_str)
        return Yield(op_type, input_types, output_types, loc_label)

    def dump(self):
        input_types_str = ", ".join([i.dump() for i in self.input_types])
        return (
            f"{self.op_type.dump()} : ({input_types_str}) -> () {self.loc_label.dump()}"
        )


@with_assert
class CallFunc(Operation):
    def update_opd(self):
        prefix = self.opd_ids[0].split(":")[0]
        number = len(self.output_types)
        self.opd_ids = [f"{prefix}:{i}" for i in range(number)]
        print(self.dump())

    def dump(self):
        input_types_str = Type.dump_type_list(self.input_types, force_list=True)
        output_types_str = Type.dump_type_list(self.output_types)
        loc_label_str = self.loc_label.dump()

        prefix = self.opd_ids[0].split("#")[0]
        number = len(self.opd_ids)
        opd_str = prefix if number == 1 else f"{prefix}:{number}"

        return f"{opd_str} = {self.op_type.dump()} : {input_types_str} -> {output_types_str} {loc_label_str}"


class GroupOp(Operation):
    """
     %6:2 = "tpu.Group"(%arg0) ({
    }) {...} : (tensor<1x3x384x288xf32, 4410114048 : i64>) -> (tensor<1x64x96x72xf32, 4411441152 : i64>, tensor<1x64x96x72xf32, 4413210624 : i64>) loc(#loc1666)
    """

    def __init__(
        self,
        opd_id: str,
        op_type: OperationType,
        ops: List[Operation],
        attrs: Attributes,
        input_types: List[Type],
        output_types: List[Type],
        loc_label: LocLabel,
    ) -> None:
        assert isinstance(opd_id, str)
        assert ":" not in opd_id

        if len(output_types) == 1:
            opd_ids = [opd_id]
        else:
            opd_ids = [f"{opd_id}#{i}" for i in range(len(output_types))]

        super().__init__(opd_ids, op_type, input_types, output_types, loc_label, attrs)
        self.ops = ops

    def dump_head(self):
        prefix = self.opd_ids[0].split("#")[0]
        number = len(self.opd_ids)
        op_id_str = f"{prefix}:{number}"
        return f"{op_id_str} = {self.op_type.dump()} ({{"

    def dump_tail(self):
        attrs_str = self._attrs.dump()
        input_types_str = Type.dump_type_list(self.input_types, force_list=True)
        output_types_str = Type.dump_type_list(self.output_types)
        loc_label_str = self.loc_label.dump()
        return (
            f"}}) {attrs_str} : {input_types_str} -> {output_types_str} {loc_label_str}"
        )

    def dump(self):
        head = self.dump_head()
        tail = self.dump_tail()
        ops_str = "\n".join([i.dump() for i in self.ops])
        ops_str = textwrap.indent(ops_str, SPACE)
        return f"{head}\n{ops_str}\n{tail}"


class Func(Node):
    """
    func.func @main(%arg0: tensor<1x3x384x288xf32> loc(unknown)) -> tensor<1x133x96x72xf32, 4418519040 : i64> {
                   argname;         type;               loc                     output_types
        ...
    } loc(#loc)

    """

    # input: List[Operation]
    # dtype/result : List[Dtype]

    # List[Operation]
    # loc: Location
    def __init__(
        self,
        name,
        input_names: List[str],
        input_types: List[Type],
        input_locs: List[LocLabel],
        output_types: List[Type],
        attr: Attributes,
        ops: List[Operation],
    ) -> None:
        super().__init__()
        self.name = name
        self.input_names = input_names
        self.input_locs = input_locs
        self.input_types = input_types
        self.output_types = output_types
        self.attr = attr
        self.ops = ops

    def erase_unused_op(self):
        counter = Counter()  # use of opd
        opdid2index = {}

        for i, op in enumerate(self.ops):
            if op.erased:
                continue

            if not op.op_type.isa("func.return"):
                for opd_id in op.opd_ids:  # can be erased
                    opdid2index[opd_id] = i
                    counter[opd_id] = 0

            if not op.op_type.isa(
                "top.Input", "top.Weight", "top.None"
            ):  # no use of other data
                for opd in op.op_type.unique_opds:
                    counter[opd] -= 1

        less = counter.most_common(1)
        while (
            len(less) > 0 and less[0][1] == 0
        ):  # while the count of some opd is 0, that operation can be erased
            for opd, count in counter.most_common():
                if count >= 0:
                    index = opdid2index[opd]
                    op = self.ops[index]
                    all_zero = True
                    for opd_id in op.opd_ids:
                        if counter[opd_id] != 0:
                            all_zero = False
                            break
                    if all_zero:
                        op.erase()
                        for opd in op.op_type.unique_opds:
                            counter[opd] += 1
                        for opd_id in op.opd_ids:
                            counter.pop(opd_id)
            less = counter.most_common(1)

    def align_input(self):
        input_types = []

        for op in self.ops:
            if op.op_type.isa("top.Input"):
                assert len(op.input_types) == 1
                op.op_type.opds = [f"%arg{len(input_types)}"]
                input_types.extend(op.input_types)

        self.input_types = input_types
        self.input_names = [f"%arg{i}" for i in range(len(input_types))]
        self.input_locs = [self.input_locs[0]] * len(input_types)

    @staticmethod
    def parse_func_inputs(func_inputs_str: str):
        """
        %arg0: tensor<1x3x384x288xf32> loc(unknown)
        %arg0: tensor<1x1600x4xf32, 4433473536 : i64> loc("p2o.Concat.30_Concat"), %arg1: tensor<1x300x2xsi32, 4433510400 : i64> loc("p2o.helper.concat.1_Concatgather_nd_0.tmp_0_GatherND_si32")
        """
        start = func_inputs_str.find("%")
        arg_names = []
        arg_types = []
        arg_locs = []
        c = 0
        while start >= 0:
            c += 1

            arg_name_end_idx = func_inputs_str.find(":", start)
            arg_name = func_inputs_str[start:arg_name_end_idx]
            tensor_start_idx = arg_name_end_idx + 1
            tensor_end_idx = func_inputs_str.find("loc", arg_name_end_idx)
            tensor_str = func_inputs_str[tensor_start_idx:tensor_end_idx].strip()
            arg_type = Type.parse(tensor_str)
            arg_names.append(arg_name)
            arg_types.append(arg_type)

            arg_loc_start_idx = tensor_end_idx
            arg_loc_end_idx = func_inputs_str.find(",", arg_loc_start_idx)
            if arg_loc_end_idx == -1:
                arg_loc_end_idx = None
            loc_label_str = func_inputs_str[arg_loc_start_idx:arg_loc_end_idx].strip()

            arg_locs.append(LocLabel.parse(loc_label_str))
            start = func_inputs_str.find("%", arg_loc_start_idx)

        return arg_names, arg_types, arg_locs

    def dump_head(self):
        input_inner_str = []
        for arg_name, type, loc in zip(
            self.input_names, self.input_types, self.input_locs
        ):
            input_inner_str.append(f"{arg_name}: {type.dump()} {loc.dump()}")
        input_inner_str = ", ".join(input_inner_str)
        if self.attr is None:
            return f"func.func @{self.name}({input_inner_str}) -> {Type.dump_type_list(self.output_types)} {{"
        else:
            return f"func.func @{self.name}({input_inner_str}) -> {Type.dump_type_list(self.output_types)} attributes {self.attr.dump()} {{"

    def dump_tail(self):
        return "} loc(#loc)"

    def dump(self):
        head = self.dump_head()
        tail = self.dump_tail()
        ops_str = "\n".join([i.dump() for i in self.ops if not i.erased])
        ops_str = textwrap.indent(ops_str, SPACE)
        return f"{head}\n{ops_str}\n{tail}"


class Module(Node):
    def __init__(
        self,
        name: str,
        attrs: Attributes,
        funcs: List[Func],
        sub_modules: List["Module"],
    ) -> None:
        super().__init__()
        self.name = name
        self.funcs = funcs
        self.attrs = attrs
        self.sub_module = sub_modules
        for sub in sub_modules:
            self.funcs.extend(sub.funcs)
            self.attrs.attributes.update(sub.attrs.attributes)

    def dump(self):
        """
        module @DragGan attributes {...} {
        } loc(#loc)
        """
        head = self.dump_head()
        tail = self.dump_tail()
        func_str = "\n".join([i.dump() for i in self.funcs])
        func_str = textwrap.indent(func_str, SPACE)
        return f"{head}\n{func_str}\n{tail}"

    def dump_head(self):
        return f"module @{self.name} attributes {self.attrs.dump()} {{"

    def dump_tail(self):
        return "} loc(#loc)"
