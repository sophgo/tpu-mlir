GMEM Allocation
===============

1. Purpose
-------------------------
In order to save global memory space and reuse memory space to the greatest extent, GMEM will be allocated to weight tensor first, and then allocated to all global neuron tensors according to their life cycle. In addition, allocated GMEM will be reused during the allocation process.

  .. note::

    global neuron tensor definition: the tensor that needs to be saved in GMEM after the Op operation.
    If it is a LayerGroup op, only the input/output tensor is considered as global neuron tensor.

1. Principle
-------------------------
2.1.  GMEM allocation in weight tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Iterate through all WeigthOp and allocate GMEM sequentially. Address space accumulation while 4K addresses are aligned.

2.2. GMEM allocation in global neuron tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Maximize the reuse of memory space. Allocate GMEM to all global neuron tensors according to their life cycle, and reuse the allocated GMEM during the allocation process.

a. Introduction of data structure:
    The corresponding tensor, address, size, ref_cnt (how many OPs are using this tensor) are recorded in rec_tbl at each allocation.
    The tensor and address are recorded in the auxiliary data structures hold_edges,in_using_addr respectively.

      .. code-block:: cpp

        //Value, offset, size, ref_cnt
        using gmem_entry = std::tuple<mlir::Value, int64_t, int64_t, int64_t>;
        std::vector<gmem_entry> rec_tbl;
        std::vector<mlir::Value> hold_edges;
        std::set<int64_t> in_using_addr;

b. Flow description:

    * Iterate through each Op, and determine if the input tensor of the Op is in rec_tbl, if yes, then determine if ref_cnt >= 1, if still yes, ref_cnt --. This operation means that the number of references to the input tensor is reduced by one.
       If ref_cnt is equal to 0, it means that the life cycle of the tensor is over, and later tensors can reuse its address space.

    * When allocating the output tensor to each Op, we first check whether the EOL tensor address can be reused. In other words, the rec_tbl must meet the following 5 conditions before it can be reused:
        * the corresponding tensor is not in the hold_edges.
        * the address of the corresponding tensor is not in_using_addr
        * The corresponding tensor is already EOL.
        * The address space of the corresponding tensor >= the space required by the current tensor.
        * The address of the input tensor of the current Op is different from the address of the corresponding tensor (e.g., the final result of some Op operations is incorrect, except for reshapeOp).

    * Allocate GMEM to the output tensor of the current Op. Reuse it if step2 shows that it can be reused. Otherwise, open a new GMEM in ddr.

    * Adjust the lifecycle of the current Op's input tensor, check if it is in hold_edges, if yes, look in rec_tbl and check if its ref_cnt is 0, if yes, remove it from hold_edges as well as its addr from in_using_addr. This operation means that the input tensor has finished its life cycle and the address space has been released.


  .. note::

    EOL definition: end-of-life.
