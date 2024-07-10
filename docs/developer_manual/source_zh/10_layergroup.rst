LayerGroup
============

基本概念
--------------

智能深度学习处理器分为片外内存(或称Global Memory, 简称GMEM)和片内内存(或称Local Memory, 简称LMEM)。

通常片外内存非常大(比如4GB), 片内内存非常小(比如16MB)。神经网络模型的数据量和计算量

都非常大, 通常每层的OP都需要切分后放到Local Memory进行运算, 结果再保存到Global Memory。

LayerGroup就是让尽可能多的OP经过切分后能够在Local Memory执行, 而避免过多的Local和Global Memory的拷贝。

要解决的问题:
   如何使Layer数据保持在有限的Local Memory进行运算, 而不是反复进行Local与Global Memory之间的拷贝
基本思路:
   通过切Activation的N和H, 使每层Layer的运算始终在Local Memory中, 如图(:ref:`lg_slice`)

.. _lg_slice:
.. figure:: ../assets/lg_slice.png
   :height: 9.5cm
   :align: center

   网络切分举例


BackwardH
--------------

对网络进行H切分的时候, 大多数Layer输入和输出的H是一致的。但是对于Conv、Pool等等需要特别计算。

以Conv举例, 如图(:ref:`backward_h`)

.. _backward_h:
.. figure:: ../assets/lg_backward.png
   :height: 9.5cm
   :align: center

   卷积BackwardH举例


划分Mem周期
--------------

如何划分group? 首先把每一层Layer需要的lmem罗列出来, 大体可以归为三类:

1. Activation Tensor, 用于保存输入输出结果, 没有使用者后直接释放
2. Weight, 用于保存权重, 不切的情况下用完就释放; 否则一直驻留在lmem
3. Buffer, 用于Layer运算保存中间结果, 用完就释放

然后依次广度优先的方式配置id, 举例如图(:ref:`lg_lmem`)

.. _lg_lmem:
.. figure:: ../assets/lg_lmem.png
   :height: 9.5cm
   :align: center

   LMEM的ID分配


然后再配置周期, 配置方法如图(:ref:`lg_timestep`)

.. _lg_timestep:
.. figure:: ../assets/lg_timestep.png
   :height: 9.5cm
   :align: center

   TimeStep分配

关于配置周期的细节如下:

- [T2,T7], 表示在T2开始的时候就要申请lmem, 在T7结束的时候释放lmem
- w4的原始周期应该是[T5,T5], 但是被修正成[T2,T5], 因为在T2做卷积运算时w4可以被同时加载
- 当N或者H被切分时, Weight不需要重新被加载, 它的结束点会被修正为正无穷

LMEM分配
--------------

当n或h存在切分的情况下, weight常驻LMEM, 每一个切分都可以继续使用weight。

这时候会先分配weight, 如图所示(:ref:`lg_nh_alloc`)

.. _lg_nh_alloc:
.. figure:: ../assets/lg_nh_alloc.png
   :height: 9.5cm
   :align: center

   有切分情况的分配

当n和h都没有切分的情况下, weight和activation处理过程一样, 不使用时就释放。

这时候的分配过程, 如图所示(:ref:`lg_alloc`)

.. _lg_alloc:
.. figure:: ../assets/lg_alloc.png
   :height: 9.5cm
   :align: center

   无切分情况的分配

那么Lmem分配问题就可以转换成这些方块如何摆放问题(注意方块只能左右移动, 不能上下移动)。

另外lmem分配时优先不要跨bank。

目前策略是按照op顺序依次分配, 优先分配timestep长的, 次分配lmem大的。

划分最优Group
--------------

.. figure:: ../assets/lg_step.png
   :align: center

   Group流程

目前从尾部开始向头部方向划分group, 优先切N, 当N切到最小单位时还不能满足要求, 则切h。

当网络很深的时候, 因为Conv、Pool等等算子会有重复计算部分, h切的过多导致重复部分过多;

为了避免过多重复, 当backward后的layer的输入, 如果h_slice重复的部分>h/2, 则认为失败。

举例: 比如input的h = 100, 经过切分后变成2个input, h[0, 80)和h[20, 100), 则重复部分为60,
则认为失败; 2个input对应h[0, 60)和h[20, 100), 重复部分为40, 认为成功。
