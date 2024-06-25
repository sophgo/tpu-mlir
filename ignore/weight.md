# BM1684X weight摆放

* content
{:toc}

## INT8算子摆放
### 非DW Conv2D

3ic_optimize = 0

#### step 1: 原始摆放

filter(int8): `[oc, ic, kh, kw]`
bias(int32): `[oc]`
requant(int32): `[1, oc, 1, 3]`
其中3表示：multiplier + rshift + input_ZeroPoint

#### step 2: filter变化

* 非 depthwise

filter(int8): `[1, oc, CEIL(ic/IC_PARALLEL), kh*kw*IC_PARALLEL]` => `[1, oc, 1, w1]`
令：`w1 = CEIL(ic/IC_PARALLEL) * kh * kw * IC_PARALLEL`

* depthwise

仅调整shape: `[1, oc, 1, w1]`, 令：`w1 = kh*kw`

#### step 3: broadcast channel

filter(int8): `[1, NPU_NUM, CEIL(oc/NPU_NUM), w1]` => `[1, NPU_NUM, 1, w2]`
令： `w2 = CEIL(oc/NPU_NUM) * w1`,


bias(int32): `[1, NPU_NUM, 1, CEIL(oc/NPU_NUM)]` => (int8)`[1, NPU_NUM, 1, w3]`
令： `w3 = CEIL(oc/NPU_NUM) * 4`,

requant(int32): `[1, NPU_NUM, 1, CEIL(oc/NPU_NUM)*REQUANT_WSIZE]` => （int8)`[1, NPU_NUM, 1, w4]`
令：`w4 = (CEIL(oc/NPU_NUM)-1) * EU_NUM + REQUANT_WSIZE * 4`
注意是中间需要align

#### step 4: merge coeff
w维度按照requant + bias + filter合并，

coeff = `[1, NPU_NUM, 1, w5]`

* 非depthwise

令: `w5 = align(w4 + w3, EU_NUM) + w2`

* depthwise

令: `w5 = w4 + w3 + w2`

#### 举例

举例：
filter(int8): `[128, 64, 3, 3]`, bias(int32):`[128]`, requant(int32):`[1,128,1,3]`
=> filter(int8): `[1, 128, 1, 3*3*64]` = `[1, 128, 1, 576]`
=> filter(int8): `[1, 64, 1, 2*576`] = `[1, 64, 1, 1152]`
   bias(int8): `[1, 64, 1, 2*4]` = `[1, 64, 1, 8]`
   requant(int8): `[1, 64, 2, 12]` = `[1, 64, 1, 64+12]`
=> merge(int8): w = CEIL(CEIL(88,4) + 8, 64) + 1152 = 1280
=> shape(int8): `[1, 64, 1, 1280]`

### DW Conv2d

3ic_optimize = 0

#### step 1: 原始摆放

filter(int8): `[oc, 1, kh, kw]`
bias(int32): `[oc]`
requant(int32): `[1, oc, 1, 3]`
其中3表示：multiplier + rshift + input_ZeroPoint

#### step 2: filter仅调整shape

filter(int8): `[1, oc, 1, kh*kw]`

## BF16/F16算子摆放

### Conv2D

#### step 1: 原始摆放

filter(bf16): `[oc, ic, kh, kw]`
bias(f32): `[oc]`

#### step2 : 仅filter变化

filter(int8): `[1, oc, CEIL(ic/32), kh*kw*32]` => `[1, oc, 1, w1]`
令：`w1 = CEIL(ic/32) * kh * kw * 32`

bias(f32): `[1, oc, 1, 1]`

## F32算子摆放

### Conv2D

#### step 1: 原始摆放

filter(f32): `[oc, ic, kh, kw]`
bias(f32): `[oc]`

#### step 2: 仅改变Shape，不改摆放

filter(f32): `[1, oc, ic, kh*kw]`
bias(f32): `[1, oc, 1, 1]`

