# Quantization

* content
{:toc}


## 量化算法

### 对称量化

用于cv183x/cv182x/mars/bm1684，通常weight用127, Activation用128
$$
\begin{align}
Quant:\quad & i8\_value = f32\_value \times \frac{128}{threshold}\\
Dequant:\quad &f32\_value = i8\_value \times \frac{threshold}{128}
\end{align}
$$

* 在1686芯片中支持u8和i8混合运算，所以对于数值>=0的情况下，`scale = threshold / 255.0`

### 非对称量化

用于BM1686

![](./assets/quant_asym.png)

* Dequant
  $$
  \begin{align}
  公式:\quad & r = S(q-Z) \\
  S:\quad & \frac{max-min}{qmax-qmin} \\
  Z:\quad & Round(- \frac{min}{S} + qmin)
  \end{align}
  $$
  当量化到INT8时，qmax=127,qmin=-128; UINT8时，qmax=255,qmin=0

* Quant

$$
\begin{align}
公式:\quad & q = \frac{r}{S} + Z
\end{align}
$$

* 为了避免类型太过于混杂，1686非对称量化只用i8，不用u8

## 算子实现

### Add

#### 表达式

$$
Y = A + B
$$

#### 量化推导

$$
\begin{align}
float:\quad & Y = A + B \\
step 0\quad & => S_y (q_y-Z_y) = S_a(q_a-Z_a) + S_b(q_b - Z_b) \\
step 1(对称) \quad & => q_y = (q_a * M_a + q_b * M_b)_{i16} >> rshift_{i8} \\
step 1(非对称) \quad & => q_y = requant(dequant(q_a) + dequant(q_b))
\end{align}
$$



#### 各平台实现

##### cv183x/cv182x/mars

$$
q_y = (q_a * M_a + q_b * M_b)_{i16} >> rshift_{i8}
$$


##### 1684

$$
q_y = ((q_a * M_a >> shift_a)_{i8} + (q_b * M_b >> shift_b)_{i8})_{i8}
$$


##### 1686

$$
q_y = requant(dequant(q_a) + dequant(q_b))
$$




### InnerProduct

#### 表达式

$$
Y = X\times W + B
$$

#### 量化推导

$$
\begin{align}

float:\quad & Y = X\times W + B \\
step 0\quad & => S_y(q_y-Z_y) = S_x(q_x-Z_x)\times S_wq_w + B \\
step 1\quad & => q_y - Z_y = S_1(q_x-Z_x)\times q_w + B_1 \\
step 2\quad & => q_y - Z_y = S_1 q_x\times q_w  + B_2 \\
step 3\quad & => q_y = S_3 (q_x \times q_w + B_3) + Z_{y} \\
step 4\quad & => q_y = (q_x \times q_w + b_{i32}) * M_{i32} >> rshift_{i8} + Z_{y}

\end{align}
$$


#### 各平台实现

##### cv183x/cv182x/mars

$$
y_{i8} = ((x_{i8}\times w_{i8})_{i32} + b_{i32}) * M_{i32} >> rshift_{i8}
$$

##### bm1684

$$
y_{i8} = ((x_{i8}\times w_{i8})_{i16} + b_{i16}) >> rshift_{i8}
$$

##### bm1686

$$
\begin{align}
y_{i8} = & (x_{i8}\times w_{i8})_{i32} + b_{i32}) * M_{i32} >> rshift_{i8} + z_{i8} \\
分两个算子实现:\quad& \\
算子1:\quad &((x_{i8}\times w_{i8})_{i32} + b_{i32})_{i32} \\
算子2:\quad & * M_{i32} >> rshift_{i8} + z_{i8}
\end{align}
$$



### Convolution

#### 表达式

$$
Y = X_{(n,ic,ih,iw)}\times K_{(oc,ic,kh,kw)} + B_{(1,oc,1,1)}
$$

#### 量化推导

略 （与InterProduct相同）

#### 各平台实现

##### cv183x/cv182x/mars

perchannel量化，其中Multiplier/Rshift/Bias会合并到一个operand里面，按照(1,oc,1,9)格式摆放，4字节bias + 4字节multiplier + 1字节rshift
$$
y_{i8} = ((x_{i8}\times k_{i8})_{i32}+b_{i32})\times M_{i32}^{oc} >> rshift_{i8}^{oc}
$$

##### bm1684

$$
y_{i8} = ((x_{i8}\times k_{i8})_{i32}+b_{i32})>> rshift_{i8}^{oc}
$$

##### bm1686

weight当前采用perchannel对称量化，activation用perlayer非对称量化
$$
y_{i8} = (x_{i8}\times (k_{i8}-z_{k}^{oc}))_{i32}+b_{i32})\times M_{i32}^{oc} >> rshift_{i8}^{oc} + z_{yi8}
$$

