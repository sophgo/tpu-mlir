用PPL写后端算子
=========================

PPL 是基于 C/C++ 语法扩展的、针对 TPU 编程的专用编程语言 (DSL)。开发者可以通过 PPL 在 TPU-MLIR 中编写后端算子。本章节以 ``fattention`` 算子为例，介绍如何编写后端算子，以及 PPL 代码是如何被编译和使用的。

PPL 后端算子的实现位于 ``tpu-mlir/lib/PplBackend/src`` 目录；如果是发布包，则在 TPU-MLIR 发布包的 ``PplBackend/src`` 目录。有关如何编写 PPL 源码的详细信息，请参考 ``tpu-mlir/third_party/ppl/doc`` 中的文档。

如何编写和调用后端算子
----------------------------

第一步：实现两个源码文件

需要创建两个源码文件，一个是设备端的 ``pl`` 源码，另一个是主机端的 ``cpp`` 源码。以 ``fattention`` 为例，文件名分别为：

- ``fattention.pl``：实现 ``flash_attention_gqa_f16`` 和 ``flash_attention_gqa_bf16`` 两个 kernel 接口。
- ``fattention.cpp``：实现 ``api_fattention_global`` 函数以调用这些 kernel 接口。

第二步：调用 Kernel 接口

在 ``FAttention.cpp`` 的 ``void tpu::FAttentionOp::codegen_global_bm1684x()`` 函数中，调用 ``api_fattention_global``，代码如下：

.. code-block:: cpp

    BM168x::call_ppl_global_func("api_fattention_global", &param, sizeof(param),
                                 input_spec->data(), output_spec->data());

如果该算子支持局部执行，则实现 ``api_xxxxOp_local``，并使用 ``BM168x::call_ppl_local_func`` 进行调用。

以上便完成了后端算子的实现。

PPL 集成到 TPU-MLIR 的流程
----------------------------

1. 将 PPL 编译器精简后放入 ``third_party/ppl`` 目录，并更新 PPL 编译器，参考该目录下的 README.md 文件。
2. 在 ``model_deploy.py`` 中集成 PPL 源码编译，流程如图所示：

.. _ppl_flow:
.. figure:: ../assets/ppl_flow.png
   :height: 9.5cm
   :align: center

   PPL Workflow
