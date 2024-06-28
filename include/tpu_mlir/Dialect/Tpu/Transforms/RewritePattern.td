#ifndef REWRITE_PATTERN
#define REWRITE_PATTERN

include "mlir/IR/PatternBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "tpu_mlir/Dialect/Tpu/IR/TpuOps.td"

def HasOneUse: Constraint<CPred<"$_self.hasOneUse()">, "has one use">;
def IsUniformQuantizedType:Constraint<CPred
        <"isa<quant::UniformQuantizedType>(cast<ShapedType>($0.getType()).getElementType())">, "is UniformQuantizedType">;

def IsCalibratedQuantizedType:Constraint<CPred
          <"$_self.getType().cast<RankedTensorType>().getElementType().isa<quant::CalibratedQuantizedType>()">,"is CalibratedQuantizedType">;

def IsCaliF16orBF16: Constraint<Or<[CPred<"$_self.getType().cast<RankedTensorType>().getElementType().cast<quant::CalibratedQuantizedType>().getExpressedType().isF16()">,
                                CPred<"$_self.getType().cast<RankedTensorType>().getElementType().cast<quant::CalibratedQuantizedType>().getExpressedType().isBF16()">]>>;
def IsWeightOp: Constraint<CPred<"isa<top::WeightOp>($0.getDefiningOp())">, "is weightOp">;
def createSplitQuantizedMLP : NativeCodeCall<"tpu_mlir::tpu::createSplitQuantizedMLP($_builder, $0.getDefiningOp(), $1)">;
def SplitQuantizedMLPPattern :  Pat<(Tpu_MatMulOp
                                          (Tpu_LutOp
                                            (Tpu_MatMulOp:$result $arg0, $arg1, $arg2, BoolAttr:$left_transpose2, BoolAttr:$right_transpose2,
                                              BoolAttr:$output_transpose2, BoolAttr:$hdim_is_batch2, BoolAttr:$keep_dims2,
                                              BoolAttr:$do_relu2, F64Attr:$relu_limit2, I64ArrayAttr:$multipliers2, I64ArrayAttr:$rshifts2,
                                              I64Attr:$right_zp2, I64Attr:$input_zp2, Tpu_RequantModeAttr:$quant_mode2,
                                              I64Attr:$left_reuse2, $arg3, $arg4, $arg5),
                                            $arg6,
                                            $arg7),
                                          $arg8, $arg9, BoolAttr:$left_transpose, BoolAttr:$right_transpose,
                                          BoolAttr:$output_transpose, BoolAttr:$hdim_is_batch, BoolAttr:$keep_dims,
                                          BoolAttr:$do_relu, F64Attr:$relu_limit, I64ArrayAttr:$multipliers, I64ArrayAttr:$rshifts,
                                          I64Attr:$right_zp, I64Attr:$input_zp, Tpu_RequantModeAttr:$quant_mode,
                                          I64Attr:$left_reuse, $arg10, $arg11, $arg12),
                                      (createSplitQuantizedMLP $result, $arg0),
                                      [(HasOneUse:$arg0),
                                      (IsUniformQuantizedType $arg0),
                                      (IsWeightOp $arg1)]
                                      >;

def SplitMixedQuantizedMLPPattern :  Pat<(Tpu_MatMulOp:$result1
                                            (Tpu_CastOp
                                              (Tpu_LutOp
                                                (Tpu_MatMulOp:$result $arg0, $arg1, $arg2, BoolAttr:$left_transpose2, BoolAttr:$right_transpose2,
                                                  BoolAttr:$output_transpose2, BoolAttr:$hdim_is_batch2, BoolAttr:$keep_dims2,
                                                  BoolAttr:$do_relu2, F64Attr:$relu_limit2, I64ArrayAttr:$multipliers2, I64ArrayAttr:$rshifts2,
                                                  I64Attr:$right_zp2, I64Attr:$input_zp2, Tpu_RequantModeAttr:$quant_mode2,
                                                  I64Attr:$left_reuse2, $arg3, $arg4, $arg5),
                                                $arg6,
                                                $arg7), $extra_input, $ginfo, $with_scale, $round_mode),
                                          $arg8, $arg9, BoolAttr:$left_transpose, BoolAttr:$right_transpose,
                                          BoolAttr:$output_transpose, BoolAttr:$hdim_is_batch, BoolAttr:$keep_dims,
                                          BoolAttr:$do_relu, F64Attr:$relu_limit, I64ArrayAttr:$multipliers, I64ArrayAttr:$rshifts,
                                          I64Attr:$right_zp, I64Attr:$input_zp, Tpu_RequantModeAttr:$quant_mode,
                                          I64Attr:$left_reuse, $arg10, $arg11, $arg12),
                                      (createSplitQuantizedMLP $result, $arg0),
                                      [(HasOneUse:$arg0),
                                      (IsUniformQuantizedType $arg0),
                                      (IsCalibratedQuantizedType:$result1),
                                      (IsCaliF16orBF16:$result1),
                                      (IsWeightOp $arg1)]
                                      >;
#endif
