Quantization
============

The theory of quantization is based on: Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference

Paper link: https://arxiv.org/abs/1712.05877

This chapter introduces the quantization design of TPU-MLIR, focusing on the application of the paper in practical quantization.

.. _quantization:

Basic Concepts
--------------

INT8 quantization is divided into symmetric and asymmetric quantization. Symmetric quantization is a special case of asymmetric quantization, and usually, the performance of the former will be better than the latter, while the accuracy is in contrast.

Asymmetric Quantization
~~~~~~~~~~~~~~~~~~~~~~~

.. _asym_quant:
.. figure:: ../assets/quant_asym.png
   :height: 9.5cm
   :align: center

   Asymmetric quantization

As shown in the figure (:ref:`asym_quant`), asymmetric quantization is actually the fixed-pointing of values in the range [min,max] to the interval [-128, 127] or [0, 255].

The quantization formula from int8 to float is:

.. math::

   r &= S(q-Z) \\
   S &= \frac{max-min}{qmax-qmin} \\
   Z &= Round(- \frac{min}{S} + qmin)

where r is the real value of type float and q is the quantized value of type INT8 or UINT8.

S denotes scale, which is float; Z is zeropoint, which is of type INT8.

When quantized to INT8, qmax=127,qmin=-128, and for UINT8, qmax=255,qmin=0.

The quantization formula from float to INT8 is:

.. math::

   q = \frac{r}{S} + Z

Symmetric Quantization
~~~~~~~~~~~~~~~~~~~~~~~

Symmetric quantization is a special case of asymmetric quantization when Z=0. The formula is:

.. math::

   i8\_value &= f32\_value \times \frac{128}{threshold} \\
   f32\_value &= i8\_value \times \frac{threshold}{128}

The range of Tensor is [-threshold, threshold].

For activation, usually :math:`S = threshold / 128`.

For weight, usually :math:`S = threshold / 127`.

In the case of UINT8, the Tensor range is [0, threshold], at this time :math:`S = threshold/ 255.0`.


Scale Conversion
----------------

The formula in the paper:

.. math::

   M = 2^{-n}M_0, where the range of M_0 is [0.5,1], and n is a non-negative number

In other words, it is the floating point Scale, which can be converted to Multiplier and rshift:

.. math::

   Scale = \frac{Multiplier}{2^{rshift}}

For example:

.. math::

   &y = x \times 0.1234 \\
   &=> y = x \times 0.9872 \times 2^{-3} \\
   &=> y = x \times (0.9872 \times 2^{31}) \times 2^{-34} \\
   &=> y = x \times \frac{2119995857}{1 \ll 34} \\
   &=> y = (x \times 2119995857) \gg 34

The higher the number of bits supported by Multiplier, the closer to Scale it will be, but that leads to worse performance. Therefore, generally, the hardware will use a 32-bit or 8-bit Multiplier.

Quantization derivation
------------------------

We can use quantization formulas and derive quantization for different OPs to get their corresponding INT8 calculations.

Both symmetric and asymmetric are used for Activation, and for weights generally only symmetric quantization is used.

.. _conv_quant:

Convolution
~~~~~~~~~~~~

The abbreviation of Convolution: :math:`Y = X_{(n,ic,ih,iw)}\times W_{(oc,ic,kh,kw)} + B_{(1,oc,1,1)}`.

Substitute it into the int8 quantization formula, the derivation is as follows:

.. math::

   float:\quad & Y = X\times W + B \\
   step 0\quad & => S_y(q_y-Z_y) = S_x(q_x-Z_x)\times S_wq_w + B \\
   step 1\quad & => q_y - Z_y = S_1(q_x-Z_x)\times q_w + B_1 \\
   step 2\quad & => q_y - Z_y = S_1 q_x\times q_w  + B_2 \\
   step 3\quad & => q_y = S_3 (q_x \times q_w + B_3) + Z_{y} \\
   step 4\quad & => q_y = (q_x \times q_w + b_{i32}) * M_{i32} >> rshift_{i8} + Z_{y}


In particular, for asymmetric quantization, Pad is filled with Zx.

In the symmetric case, Pad is filled with 0 (both Zx and Zy are 0).

In PerAxis (or PerChannal) quantization, each OC of Filter will be quantized, and the derivation formula will remain unchanged, but there will be OC Multiplier and rshift.


InnerProduct
~~~~~~~~~~~~

Expression and derivation are the same as (:ref:`conv_quant`).


Add
~~~~~~~~~~~~

The expression for addition is: :math:`Y = A + B`

Substitute it into the int8 quantization formula, the derivation is as follows:

.. math::

   float:\quad & Y = A + B \\
   step 0\quad & => S_y (q_y-Z_y) = S_a(q_a-Z_a) + S_b(q_b - Z_b) \\
   step 1(Symmetric) \quad & => q_y = (q_a * M_a + q_b * M_b)_{i16} >> rshift_{i8} \\
   step 1(Asymmetric) \quad & => q_y = requant(dequant(q_a) + dequant(q_b))

The way to implement Add with Tensor Computing Processor is related to specific processor instructions.

The symmetric method here is to use INT16 as the intermediate buffer.

The asymmetric method is to first de-quantize into the float, do the addition and then re-quantize into INT8.


AvgPool
~~~~~~~~~~~~

The expression of average pooling can be abbreviated as: :math:`Y_i = \frac{\sum_{j=0}^{k}{(X_j)}}{k}, k = kh \times kw`.

Substitute it into the int8 quantization formula, the derivation is as follows:

.. math::

   float:\quad & Y_i = \frac{\sum_{j=0}^{k}{(X_j)}}{k} \\
   step0:\quad & => S_y(y_i - Z_y) = \frac{S_x\sum_{j=0}^{k}(x_j-Z_x)}{k}\\
   step1:\quad & => y_i = \frac{S_x}{S_yk}\sum_{j=0}^{k}(x_j-Z_x) + Z_y \\
   step2:\quad & => y_i = \frac{S_x}{S_yk}\sum_{j=0}^{k}(x_j) - (Z_y - \frac{S_x}{S_y}Z_x) \\
   step3:\quad & => y_i = (Scale_{f32}\sum_{j=0}^{k}(x_j) - Offset_{f32})_{i8} \\
               & Scale_{f32} = \frac{S_x}{S_yk}, Offset_{f32} = Z_y - \frac{S_x}{S_y}Z_x


LeakyReLU
~~~~~~~~~~~~

The expression of LeakyReLU can be abbreviated as: :math:`Y = \begin{cases} X, if X \geq 0\\ \alpha X, if X < 0 \end{cases}`

Substitute it into the int8 quantization formula, the derivation is as follows:

.. math::

   float:\quad & Y = \begin{cases} X, if \ X \geq 0\\ \alpha X, if \ X < 0 \end{cases} \\
   step0:\quad & => S_y (q_y - Z_y) = \begin{cases} S_x(q_x - Z_x), if \ q_x \geq 0\\ \alpha S_x (q_x - Z_x), if \ q_x < 0 \end{cases} \\
   step1:\quad & => q_y = \begin{cases} \frac{S_x}{S_y}(q_x - Z_x) + Z_y, if \ q_x \geq 0\\ \alpha \frac{S_x}{S_y} (q_x - Z_x) + Z_y, if \ q_x < 0 \end{cases}

In INT8 symmetric quantization: :math:`S_y=\frac{threshold_y}{128}, S_x=\frac{threshold_x}{128}`. In INT8 asymmetric quantization: :math:`S_y = \frac{max_y ⁡- min_y}{255}, S_x = \frac{max_x ⁡- min_x}{255}`. After BackwardCalibration, :math:`max_y = max_x, min_y = min_x, threshold_y = threshold_x`, so Sx/Sy = 1.

.. math::

   step2:\quad & => q_y = \begin{cases} (q_x - Z_x) + Z_y,  if \ q_x \geq 0\\ \alpha (q_x - Z_x) + Z_y, if \ q_x < 0 \end{cases} \\
   step3:\quad & => q_y = \begin{cases} q_x - Z_x + Z_y,  if \ q_x \geq 0\\ M_{i8} >> rshift_{i8} (q_x - Z_x) + Z_y, if \ q_x < 0 \end{cases}

In the symmetric case, both Zx and Zy are 0.

Pad
~~~~~~~~~~~~

The expression of Pad can be abbreviated as: :math:`Y = \begin{cases} X, \ origin\ location \\ value, \ padded\ location \end{cases}`

Substitute it into the int8 quantization formula, the derivation is as follows:

.. math::
   float:\quad & Y = \begin{cases} X, \ origin\ location \\ value, \ padded\ location \end{cases} \\
   step0:\quad & => S_y (q_y - Z_y) = \begin{cases} S_x (q_x - Z_x), \ origin\ location \\ value, \ padded\ location \end{cases} \\
   step1:\quad & => q_y = \begin{cases} \frac{S_x}{S_y} (q_x - Z_x) + Z_y, \ origin\ location \\ \frac{value}{S_y} + Z_y, \ padded\ location \end{cases}

After BackwardCalibration, :math:`max_y = max_x,  min_y = min_x, threshold_y = threshold_x`, so Sx/Sy = 1。

.. math::
   step2:\quad & => q_y = \begin{cases} (q_x - Z_x) + Z_y, \ origin\ location \\ \frac{value}{S_y} + Z_y, \ padded\ location \end{cases}

In the symmetric case, both Zx and Zy are 0, so the padded value is round(value/Sy). When asymmetric quantization, the padded value is round(value/Sy + Zy)。


PReLU
~~~~~~~~~~~~
The expression of PReLU can be abbreviated as: :math:`Y_i = \begin{cases} X_i, if \ X_i \geq 0\\ \alpha_i X_i, if \ X_i < 0 \end{cases}`

Substitute it into the int8 quantization formula, the derivation is as follows:

.. math::
   float:\quad & Y_i = \begin{cases} X_i, if \  X_i \geq 0\\ \alpha_i X_i, if \ X_i < 0 \end{cases} \\
   step0:\quad & => S_y (y_i - Z_y) = \begin{cases} S_x (x_i - Z_x), if \ x_i \geq 0\\ S_{\alpha}q_{\alpha_i}S_x (x_i - Z_x), if \ x_i < 0 \end{cases} \\
   step1:\quad & => y_i = \begin{cases} \frac{S_x}{S_y} (x_i - Z_x) + Z_y, if \ x_i \geq 0\\ S_{\alpha}q_{\alpha_i}\frac{S_x}{S_y} (x_i - Z_x) + Z_y, if \ x_i < 0 \end{cases} \\

After BackwardCalibration, :math:`max_y = max_x,  min_y = min_x, threshold_y = threshold_x`, so Sx/Sy = 1。

.. math::
   step2:\quad & => y_i = \begin{cases} (x_i - Z_x) + Z_y, if \ x_i \geq 0\\ S_{\alpha}q_{\alpha_i}(x_i - Z_x) + Z_y, if \ x_i < 0 \end{cases} \\
   step3:\quad & => y_i = \begin{cases} (x_i - Z_x) + Z_y, if \ x_i \geq 0\\ q_{\alpha_i} * M_{i8} (x_i - Z_x) >> rshift_{i8} + Z_y, if \ x_i < 0 \end{cases} \\

There are oc Multipliers and 1 rshift. When symmetric quantization, Zx and Zy are both 0.
