# 已验证通过的 CUDA 算子测试命令 (36个)

测试环境: bm1684x, f32, --cuda

## 激活函数

```
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Erf"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Exp"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Elu"
```

## 逐元素运算

```
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Clip"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "DivConst"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Max"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Min"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Mish"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Pow"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Sign"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Softplus"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Softsign"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Swish"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Tan"
```

## 三角函数

```
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Sin"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Sinh"
```

## 矩阵运算

```
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Einsum"
```

## 数据操作

```
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "MaskedFill"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "ScatterElements"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "ScatterND"
```

## 形状操作

```
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Pack"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "ShuffleChannel"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "SliceAxis"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "StridedSlice"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "SwapChannel"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Unpack"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Shape"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "ShapeSlice"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Split"
```

## 数学函数

```
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Sqrt"
```

## 其他

```
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "MatchTemplate"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "MeanStdScale"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "SelectiveScan"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "TopK"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Trilu"
python test_onnx.py --chip bm1684x --mode f32 --cuda --case "Where"
```
