# BM1686 weight摆放

## INT8算子摆放
### Conv2D

depthwise = false, 3ic_optimize = 0

#### step 1: 原始摆放

filter(int8): `[oc, ic, kh, kw]`
bias(int32): `[oc]`
requant(int32): `[1, oc, 1, 3]`
其中3表示：multiplier + rshift + input_ZeroPoint

#### step 2: filter变化

filter(int8): `[1, oc, UP(ic/64), kh*kw*64]` => `[1, oc, 1, w1]`
令：`w1 = UP(ic/64) * kh * kw * 64`

#### step 3: broadcast channel

filter(int8): `[1, 64, UP(oc/64), w1]` => `[1, 64, 1, w2]`
令： `w2 = UP(oc/64) * w1`

bias(int32): `[1, 64, 1, UP(oc/64)]` => `[1, 64, 1, w3]`
令： `w3 = UP(oc/64)`

requant(int32): `[1, 64, 1, UP(oc/64)*3]` => `[1, 64, 1, w4]`
令： `w4 = EU_ALGIN(UP(oc/64)*3)`

#### step 4: merge coeff
w维度按照requant + bias + filter合并，

coeff = `[1, 64, 1, w5]`
令: `w5 = w4 + w3 + w2`

## F32算子摆放

### Conv2D

#### step 1: 原始摆放

filter(f32): `[oc, ic, kh, kw]`
bias(f32): `[oc]`

#### step 2: 仅改变Shape，不改摆放

filter(f32): `[1, oc, ic, kh*kw]`
bias(f32): `[1, oc, 1, 1]`

