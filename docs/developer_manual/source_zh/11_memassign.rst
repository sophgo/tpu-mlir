GMEM分配
============

目的
-------------------------
为了节约global memory空间,最大程度复用内存空间,分配顺序:weight tensor、根据生命周期给全部global neuron tensor分配gmem,在分配过程中会复用已分配gmem

  .. note::

    global neuron tensor定义: 在Op运算结束后需要保存在gmem的tensor.
    如果是layer Group,只有layer Group的input/output tensor
    属于global neuron tensor.

原理
-------------------------
weight tensor分配gmem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
遍历所有WeightOp,依次分配,4K地址对齐,地址空间不断累加

global neuron tensors分配gmem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
最大可能的复用内存空间,根据生命周期给全部global neuron tensor分配,在分配过程中会复用已分配gmem。

a. 数据结构介绍

    每次分配时把对应的tensor, address, size, ref_cnt(这个tensor有几个OP使用)记录在rec_tbl.
    同时将tensor, address记录在辅助数据结构hold_edges,in_using_addr中

      .. code-block:: cpp

        //Value, offset, size, ref_cnt
        using gmem_entry = std::tuple<mlir::Value, int64_t, int64_t, int64_t>;
        std::vector<gmem_entry> rec_tbl;
        std::vector<mlir::Value> hold_edges;
        std::set<int64_t> in_using_addr;

b. 流程介绍

    * **遍历每个Op, 在遍历Op时,判断Op的输入tensor是否位于rec_tbl**

      **中, 如果yes,判断ref_cnt是否>=1, 如果yes,则ref_cnt--,表示输**

      **入tensor的引用数降低1个。**

      如果ref_cnt等于0,表示生命周期已结束,后面的tensor可以复用它的地址空间。

    * **在给每个Op的output tensor分配时,先check是否可以复用EOL的**

      **tensor地址,check思路,遍历rec_tbl, 需要同时满足如下5个条件才**

      **能reuse:**

        * 对应的tensor不在hold_edges内
        * 对应tensor的地址不在in_using_addr内
        * 对应tensor已EOL
        * 对应tensor的地址空间>=当前tensor所需空间
        * 当前OP的输入tensor地址不能与对应tensor的地址相同(某些Op最终运算结果不正确,reshapeOP例外)

    * 给当前Op的output tensor分配gmem, 如果step2显示可以reuse,就reuse.否则在ddr中新开辟gmem.

    * 调整当前Op的input tensor的生命周期,确认它是否位于hold_edges内, 如果yes, 则在rec_tbl中寻找,检查它的ref_cnt是否为0,
      如果yes,则把它从hold_edges中删除,并且把它的addr从in_using_addr中删除,意味着这个input tensor生命周期已结束,地址空间已释放。


    .. image:: ../assets/gmem.png


  .. note::

    EOL定义: end of life.
