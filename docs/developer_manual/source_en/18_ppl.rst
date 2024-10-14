Implementing Backend Operators with PPL
=========================================

PPL (Programming Language for TPUs) is a domain-specific programming language (DSL) based on C/C++ syntax extensions, designed for programming Tensor Processing Units (TPUs). This chapter demonstrates how to implement backend operators in PPL using the ``fattention`` operator as an example, illustrating the compilation and utilization of PPL code within TPU-MLIR.

The implementation of PPL backend operators can be found in the ``tpu-mlir/lib/PplBackend/src`` directory. For release packages, it will be located in the ``PplBackend/src`` directory of the TPU-MLIR release package. For detailed instructions on writing PPL source code, refer to the documentation in ``tpu-mlir/third_party/ppl/doc``.

How to Write and Call Backend Operators
-----------------------------------------

### Step 1: Implement Two Source Files

You need to create two source files: one for the device-side ``pl`` code and one for the host-side ``cpp`` code. For the ``fattention`` example, these files are:

- ``fattention.pl``: Implements the ``flash_attention_gqa_f16`` and ``flash_attention_gqa_bf16`` kernel interfaces.
- ``fattention.cpp``: Implements the ``api_fattention_global`` function to call these kernel interfaces.

### Step 2: Call the Kernel Interface

In the function ``void tpu::FAttentionOp::codegen_global_bm1684x()`` within ``FAttention.cpp``, call ``api_fattention_global`` as follows:

.. code-block:: cpp

    BM168x::call_ppl_global_func("api_fattention_global", &param, sizeof(param),
                                  input_spec->data(), output_spec->data());

If the operator supports local execution, implement ``api_xxxxOp_local`` and call it using ``BM168x::call_ppl_local_func``.

This completes the implementation of the backend operator.

PPL Workflow in TPU-MLIR
-------------------------

1. Place the PPL compiler in the ``third_party/ppl`` directory and update it by referring to the README.md file in this directory.
2. Integrate the PPL source code compilation in ``model_deploy.py``. The process is illustrated in the following diagram:

.. _ppl_flow:
.. figure:: ../assets/ppl_flow.png
   :height: 9.5cm
   :align: center

   PPL Workflow
