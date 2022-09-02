Overall Design
==============

.. _dialect:

Layered
------------

TPU-MLIR treats the compilation process of the network model in two layers.

Top Dialect
   Chip-independent layer, including graph optimization, quantization and inference, etc.
Tpu Dialect
   Chip-related layer, including weight reordering, operator slicing, address assignment, inference, etc.

The overall flow is shown in the (:ref:`main_flow`) diagram, where the model is gradually converted into final instructions by Passes. Here is a detailed description of what functions each Pass does in the Top layer and the Tpu layer. The following chapters will explain the key points of each Pass in detail.

.. _main_flow:
.. figure:: ../assets/flow.png
   :align: center

   TPU-MLIR overall process



.. _top pass:

Top Pass
------------

Canonicalize
    Graph optimization related to specific OP, such as merging relu into conv, shape merge, etc.
Calibration
    According to the calibration table, insert min and max for each OP for subsequent quantization. Insert threshold for symmetric quantization.
Lowering
    Lower the OP to the tpu layer according to the quantization type. Supported types are F32/F16/BF16/INT8 symmetric/INT8 asymmetric.


.. _tpu pass:

Tpu Pass
------------

Canonicalize
   Graph optimization related to specific OP, such as merging of consecutive Requants, etc.
WeightReorder
   Reorder the weights of individual OP based on chip characteristics, such as filter and bias for convolution.
Subnet
   Split the network into different subnets according to TPU/CPU, if all operators are TPU, there is only one subnet.
LayerGroup
   Slice the network so that as many OPs as possible are computed consecutively in the local mem.
MemAssign
   Assign addresses to the OPs that need global mem.
CodeGen
   Use Builder module to generate the final model in flatbuffers format.
