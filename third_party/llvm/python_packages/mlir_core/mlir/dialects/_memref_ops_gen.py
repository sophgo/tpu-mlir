
# Autogenerated by mlir-tblgen; don't manually edit.

from ._ods_common import _cext as _ods_cext
from ._ods_common import extend_opview_class as _ods_extend_opview_class, segmented_accessor as _ods_segmented_accessor, equally_sized_accessor as _ods_equally_sized_accessor, get_default_loc_context as _ods_get_default_loc_context, get_op_result_or_value as _get_op_result_or_value, get_op_results_or_values as _get_op_results_or_values
_ods_ir = _ods_cext.ir

try:
  from . import _memref_ops_ext as _ods_ext_module
except ImportError:
  _ods_ext_module = None

import builtins


@_ods_cext.register_dialect
class _Dialect(_ods_ir.Dialect):
  DIALECT_NAMESPACE = "memref"
  pass


@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class AssumeAlignmentOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.assume_alignment"

  _ODS_REGIONS = (0, True)

  def __init__(self, memref, alignment, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(memref))
    attributes["alignment"] = alignment
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def memref(self):
    return self.operation.operands[0]

  @builtins.property
  def alignment(self):
    return _ods_ir.IntegerAttr(self.operation.attributes["alignment"])

  @alignment.setter
  def alignment(self, value):
    if value is None:
      raise ValueError("'None' not allowed as value for mandatory attributes")
    self.operation.attributes["alignment"] = value

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class AtomicRMWOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.atomic_rmw"

  _ODS_REGIONS = (0, True)

  def __init__(self, result, kind, value, memref, indices, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(value))
    operands.append(_get_op_result_or_value(memref))
    operands.extend(_get_op_results_or_values(indices))
    attributes["kind"] = kind
    results.append(result)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def value(self):
    return self.operation.operands[0]

  @builtins.property
  def memref(self):
    return self.operation.operands[1]

  @builtins.property
  def indices(self):
    _ods_variadic_group_length = len(self.operation.operands) - 3 + 1
    return self.operation.operands[2:2 + _ods_variadic_group_length]

  @builtins.property
  def result(self):
    return self.operation.results[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class AtomicYieldOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.atomic_yield"

  _ODS_REGIONS = (0, True)

  def __init__(self, result, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(result))
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def result(self):
    return self.operation.operands[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class CopyOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.copy"

  _ODS_REGIONS = (0, True)

  def __init__(self, source, target, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(source))
    operands.append(_get_op_result_or_value(target))
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def source(self):
    return self.operation.operands[0]

  @builtins.property
  def target(self):
    return self.operation.operands[1]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class GenericAtomicRMWOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.generic_atomic_rmw"

  _ODS_REGIONS = (1, True)

  @builtins.property
  def memref(self):
    return self.operation.operands[0]

  @builtins.property
  def indices(self):
    _ods_variadic_group_length = len(self.operation.operands) - 2 + 1
    return self.operation.operands[1:1 + _ods_variadic_group_length]

  @builtins.property
  def result(self):
    return self.operation.results[0]

  @builtins.property
  def atomic_body(self):
    return self.regions[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class LoadOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.load"

  _ODS_REGIONS = (0, True)

  def __init__(self, result, memref, indices, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(memref))
    operands.extend(_get_op_results_or_values(indices))
    results.append(result)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def memref(self):
    return self.operation.operands[0]

  @builtins.property
  def indices(self):
    _ods_variadic_group_length = len(self.operation.operands) - 2 + 1
    return self.operation.operands[1:1 + _ods_variadic_group_length]

  @builtins.property
  def result(self):
    return self.operation.results[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class AllocOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.alloc"

  _ODS_OPERAND_SEGMENTS = [-1,-1,]

  _ODS_REGIONS = (0, True)

  def __init__(self, memref, dynamicSizes, symbolOperands, *, alignment=None, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_results_or_values(dynamicSizes))
    operands.append(_get_op_results_or_values(symbolOperands))
    if alignment is not None: attributes["alignment"] = alignment
    results.append(memref)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def dynamicSizes(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 0)
    return operand_range

  @builtins.property
  def symbolOperands(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 1)
    return operand_range

  @builtins.property
  def alignment(self):
    if "alignment" not in self.operation.attributes:
      return None
    return _ods_ir.IntegerAttr(self.operation.attributes["alignment"])

  @alignment.setter
  def alignment(self, value):
    if value is not None:
      self.operation.attributes["alignment"] = value
    elif "alignment" in self.operation.attributes:
      del self.operation.attributes["alignment"]

  @alignment.deleter
  def alignment(self):
    del self.operation.attributes["alignment"]

  @builtins.property
  def memref(self):
    return self.operation.results[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class AllocaOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.alloca"

  _ODS_OPERAND_SEGMENTS = [-1,-1,]

  _ODS_REGIONS = (0, True)

  def __init__(self, memref, dynamicSizes, symbolOperands, *, alignment=None, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_results_or_values(dynamicSizes))
    operands.append(_get_op_results_or_values(symbolOperands))
    if alignment is not None: attributes["alignment"] = alignment
    results.append(memref)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def dynamicSizes(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 0)
    return operand_range

  @builtins.property
  def symbolOperands(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 1)
    return operand_range

  @builtins.property
  def alignment(self):
    if "alignment" not in self.operation.attributes:
      return None
    return _ods_ir.IntegerAttr(self.operation.attributes["alignment"])

  @alignment.setter
  def alignment(self, value):
    if value is not None:
      self.operation.attributes["alignment"] = value
    elif "alignment" in self.operation.attributes:
      del self.operation.attributes["alignment"]

  @alignment.deleter
  def alignment(self):
    del self.operation.attributes["alignment"]

  @builtins.property
  def memref(self):
    return self.operation.results[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class AllocaScopeOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.alloca_scope"

  _ODS_REGIONS = (1, True)

  def __init__(self, results_, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    results.extend(results_)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def results_(self):
    _ods_variadic_group_length = len(self.operation.results) - 1 + 1
    return self.operation.results[0:0 + _ods_variadic_group_length]

  @builtins.property
  def bodyRegion(self):
    return self.regions[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class AllocaScopeReturnOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.alloca_scope.return"

  _ODS_REGIONS = (0, True)

  def __init__(self, results_, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.extend(_get_op_results_or_values(results_))
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def results_(self):
    _ods_variadic_group_length = len(self.operation.operands) - 1 + 1
    return self.operation.operands[0:0 + _ods_variadic_group_length]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class CastOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.cast"

  _ODS_REGIONS = (0, True)

  def __init__(self, dest, source, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(source))
    results.append(dest)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def source(self):
    return self.operation.operands[0]

  @builtins.property
  def dest(self):
    return self.operation.results[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class CollapseShapeOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.collapse_shape"

  _ODS_REGIONS = (0, True)

  def __init__(self, result, src, reassociation, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(src))
    attributes["reassociation"] = reassociation
    results.append(result)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def src(self):
    return self.operation.operands[0]

  @builtins.property
  def result(self):
    return self.operation.results[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class DeallocOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.dealloc"

  _ODS_REGIONS = (0, True)

  def __init__(self, memref, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(memref))
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def memref(self):
    return self.operation.operands[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class DimOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.dim"

  _ODS_REGIONS = (0, True)

  def __init__(self, result, source, index, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(source))
    operands.append(_get_op_result_or_value(index))
    results.append(result)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def source(self):
    return self.operation.operands[0]

  @builtins.property
  def index(self):
    return self.operation.operands[1]

  @builtins.property
  def result(self):
    return self.operation.results[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class DmaStartOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.dma_start"

  _ODS_REGIONS = (0, True)

  def __init__(self, operands_, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.extend(_get_op_results_or_values(operands_))
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def operands_(self):
    _ods_variadic_group_length = len(self.operation.operands) - 1 + 1
    return self.operation.operands[0:0 + _ods_variadic_group_length]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class DmaWaitOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.dma_wait"

  _ODS_REGIONS = (0, True)

  def __init__(self, tagMemRef, tagIndices, numElements, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(tagMemRef))
    operands.extend(_get_op_results_or_values(tagIndices))
    operands.append(_get_op_result_or_value(numElements))
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def tagMemRef(self):
    return self.operation.operands[0]

  @builtins.property
  def tagIndices(self):
    _ods_variadic_group_length = len(self.operation.operands) - 3 + 1
    return self.operation.operands[1:1 + _ods_variadic_group_length]

  @builtins.property
  def numElements(self):
    _ods_variadic_group_length = len(self.operation.operands) - 3 + 1
    return self.operation.operands[2 + _ods_variadic_group_length - 1]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class ExpandShapeOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.expand_shape"

  _ODS_REGIONS = (0, True)

  def __init__(self, result, src, reassociation, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(src))
    attributes["reassociation"] = reassociation
    results.append(result)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def src(self):
    return self.operation.operands[0]

  @builtins.property
  def result(self):
    return self.operation.results[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class GetGlobalOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.get_global"

  _ODS_REGIONS = (0, True)

  def __init__(self, result, name, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    attributes["name"] = name
    results.append(result)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def result(self):
    return self.operation.results[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class GlobalOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.global"

  _ODS_REGIONS = (0, True)

  def __init__(self, sym_name, type, *, sym_visibility=None, initial_value=None, constant=None, alignment=None, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    attributes["sym_name"] = sym_name
    if sym_visibility is not None: attributes["sym_visibility"] = sym_visibility
    attributes["type"] = type
    if initial_value is not None: attributes["initial_value"] = initial_value
    if bool(constant): attributes["constant"] = _ods_ir.UnitAttr.get(
      _ods_get_default_loc_context(loc))
    if alignment is not None: attributes["alignment"] = alignment
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def sym_name(self):
    return _ods_ir.StringAttr(self.operation.attributes["sym_name"])

  @sym_name.setter
  def sym_name(self, value):
    if value is None:
      raise ValueError("'None' not allowed as value for mandatory attributes")
    self.operation.attributes["sym_name"] = value

  @builtins.property
  def sym_visibility(self):
    if "sym_visibility" not in self.operation.attributes:
      return None
    return _ods_ir.StringAttr(self.operation.attributes["sym_visibility"])

  @sym_visibility.setter
  def sym_visibility(self, value):
    if value is not None:
      self.operation.attributes["sym_visibility"] = value
    elif "sym_visibility" in self.operation.attributes:
      del self.operation.attributes["sym_visibility"]

  @sym_visibility.deleter
  def sym_visibility(self):
    del self.operation.attributes["sym_visibility"]

  @builtins.property
  def initial_value(self):
    if "initial_value" not in self.operation.attributes:
      return None
    return _ods_ir.Attribute(self.operation.attributes["initial_value"])

  @initial_value.setter
  def initial_value(self, value):
    if value is not None:
      self.operation.attributes["initial_value"] = value
    elif "initial_value" in self.operation.attributes:
      del self.operation.attributes["initial_value"]

  @initial_value.deleter
  def initial_value(self):
    del self.operation.attributes["initial_value"]

  @builtins.property
  def constant(self):
    return "constant" in self.operation.attributes

  @constant.setter
  def constant(self, value):
    if bool(value):
      self.operation.attributes["constant"] = _ods_ir.UnitAttr.get()
    elif "constant" in self.operation.attributes:
      del self.operation.attributes["constant"]

  @constant.deleter
  def constant(self):
    del self.operation.attributes["constant"]

  @builtins.property
  def alignment(self):
    if "alignment" not in self.operation.attributes:
      return None
    return _ods_ir.IntegerAttr(self.operation.attributes["alignment"])

  @alignment.setter
  def alignment(self, value):
    if value is not None:
      self.operation.attributes["alignment"] = value
    elif "alignment" in self.operation.attributes:
      del self.operation.attributes["alignment"]

  @alignment.deleter
  def alignment(self):
    del self.operation.attributes["alignment"]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class PrefetchOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.prefetch"

  _ODS_REGIONS = (0, True)

  def __init__(self, memref, indices, isWrite, localityHint, isDataCache, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(memref))
    operands.extend(_get_op_results_or_values(indices))
    attributes["isWrite"] = isWrite
    attributes["localityHint"] = localityHint
    attributes["isDataCache"] = isDataCache
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def memref(self):
    return self.operation.operands[0]

  @builtins.property
  def indices(self):
    _ods_variadic_group_length = len(self.operation.operands) - 2 + 1
    return self.operation.operands[1:1 + _ods_variadic_group_length]

  @builtins.property
  def isWrite(self):
    return _ods_ir.BoolAttr(self.operation.attributes["isWrite"])

  @isWrite.setter
  def isWrite(self, value):
    if value is None:
      raise ValueError("'None' not allowed as value for mandatory attributes")
    self.operation.attributes["isWrite"] = value

  @builtins.property
  def localityHint(self):
    return _ods_ir.IntegerAttr(self.operation.attributes["localityHint"])

  @localityHint.setter
  def localityHint(self, value):
    if value is None:
      raise ValueError("'None' not allowed as value for mandatory attributes")
    self.operation.attributes["localityHint"] = value

  @builtins.property
  def isDataCache(self):
    return _ods_ir.BoolAttr(self.operation.attributes["isDataCache"])

  @isDataCache.setter
  def isDataCache(self, value):
    if value is None:
      raise ValueError("'None' not allowed as value for mandatory attributes")
    self.operation.attributes["isDataCache"] = value

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class RankOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.rank"

  _ODS_REGIONS = (0, True)

  def __init__(self, result, memref, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(memref))
    results.append(result)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def memref(self):
    return self.operation.operands[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class ReinterpretCastOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.reinterpret_cast"

  _ODS_OPERAND_SEGMENTS = [1,-1,-1,-1,]

  _ODS_REGIONS = (0, True)

  def __init__(self, result, source, offsets, sizes, strides, static_offsets, static_sizes, static_strides, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(source))
    operands.append(_get_op_results_or_values(offsets))
    operands.append(_get_op_results_or_values(sizes))
    operands.append(_get_op_results_or_values(strides))
    attributes["static_offsets"] = static_offsets
    attributes["static_sizes"] = static_sizes
    attributes["static_strides"] = static_strides
    results.append(result)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def source(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 0)
    return operand_range[0]

  @builtins.property
  def offsets(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 1)
    return operand_range

  @builtins.property
  def sizes(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 2)
    return operand_range

  @builtins.property
  def strides(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 3)
    return operand_range

  @builtins.property
  def result(self):
    return self.operation.results[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class ReshapeOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.reshape"

  _ODS_REGIONS = (0, True)

  def __init__(self, result, source, shape, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(source))
    operands.append(_get_op_result_or_value(shape))
    results.append(result)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def source(self):
    return self.operation.operands[0]

  @builtins.property
  def shape(self):
    return self.operation.operands[1]

  @builtins.property
  def result(self):
    return self.operation.results[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class StoreOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.store"

  _ODS_REGIONS = (0, True)

  def __init__(self, value, memref, indices, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(value))
    operands.append(_get_op_result_or_value(memref))
    operands.extend(_get_op_results_or_values(indices))
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def value(self):
    return self.operation.operands[0]

  @builtins.property
  def memref(self):
    return self.operation.operands[1]

  @builtins.property
  def indices(self):
    _ods_variadic_group_length = len(self.operation.operands) - 3 + 1
    return self.operation.operands[2:2 + _ods_variadic_group_length]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class TransposeOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.transpose"

  _ODS_REGIONS = (0, True)

  def __init__(self, result, in_, permutation, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(in_))
    attributes["permutation"] = permutation
    results.append(result)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def in_(self):
    return self.operation.operands[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class ViewOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.view"

  _ODS_REGIONS = (0, True)

  def __init__(self, result, source, byte_shift, sizes, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(source))
    operands.append(_get_op_result_or_value(byte_shift))
    operands.extend(_get_op_results_or_values(sizes))
    results.append(result)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def source(self):
    return self.operation.operands[0]

  @builtins.property
  def byte_shift(self):
    return self.operation.operands[1]

  @builtins.property
  def sizes(self):
    _ods_variadic_group_length = len(self.operation.operands) - 3 + 1
    return self.operation.operands[2:2 + _ods_variadic_group_length]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class SubViewOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.subview"

  _ODS_OPERAND_SEGMENTS = [1,-1,-1,-1,]

  _ODS_REGIONS = (0, True)

  def __init__(self, result, source, offsets, sizes, strides, static_offsets, static_sizes, static_strides, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(source))
    operands.append(_get_op_results_or_values(offsets))
    operands.append(_get_op_results_or_values(sizes))
    operands.append(_get_op_results_or_values(strides))
    attributes["static_offsets"] = static_offsets
    attributes["static_sizes"] = static_sizes
    attributes["static_strides"] = static_strides
    results.append(result)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def source(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 0)
    return operand_range[0]

  @builtins.property
  def offsets(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 1)
    return operand_range

  @builtins.property
  def sizes(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 2)
    return operand_range

  @builtins.property
  def strides(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 3)
    return operand_range

  @builtins.property
  def result(self):
    return self.operation.results[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class TensorStoreOp(_ods_ir.OpView):
  OPERATION_NAME = "memref.tensor_store"

  _ODS_REGIONS = (0, True)

  def __init__(self, tensor, memref, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(tensor))
    operands.append(_get_op_result_or_value(memref))
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def tensor(self):
    return self.operation.operands[0]

  @builtins.property
  def memref(self):
    return self.operation.operands[1]
