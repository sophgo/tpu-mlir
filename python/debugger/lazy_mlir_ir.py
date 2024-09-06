import mlir
import mlir.ir as mlir_ir
from mlir.dialects.func import FuncOp
from mlir.dialects import quant
from mlir.ir import *


def parse_attribute(attr: Attribute) -> dict:
    if isinstance(attr, NamedAttribute):
        return {attr.name: parse_attribute(attr.attr)}
    if isinstance(attr, (StringAttr, IntegerAttr, BoolAttr)):
        return attr.value

    if isinstance(attr, (ArrayAttr)):
        return [parse_attribute(i) for i in attr]


ir_context = mlir.ir.Context()


def parse_mlir_type(asm: str):
    element_type = Type.parse(asm, ir_context).element_type
    if quant.UniformQuantizedType.isinstance(element_type):
        quant_type = quant.UniformQuantizedType(element_type)
        return quant_type
    if quant.CalibratedQuantizedType.isinstance(element_type):
        quant_type = quant.CalibratedQuantizedType(element_type)
        return quant_type
    return RankedTensorType.parse(asm, ir_context)
