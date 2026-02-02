Appendix.07: Visualization Tool Guidance
==================================
This chapter mainly introduces the use of the visualization tool. Currently, a web-based online visualization tool is provided for LayerGroup-related visualizations (not covered here for now) and neuron space visualization. This tool supports functionalities such as inspecting peak neuron memory usage and analyzing tensor lifetimes.

Website: https://tpu-mlir-vis-tools.github.io/demo/#/neuron

This tool does not support addr_mode modes other than auto.


Preparatory work
------------------

**Environment configuration**

First you need to refer to :ref:`Environment Setup chapter <env setup>` to complete the environment configuration, enter the Docker container of TPU-MLIR, and install tpu_mlir in it.

If you have completed the environment configuration, you can ignore this step.


**Generate final.mlir**

Before using tool, you need to generate the final.mlir file through TPU-MLIR, refer to :ref:`Compile the ONNX model chapter <onnx to bmodel>` to generate the bmodel file from the model.

You need to use the following two commands:

.. code-block:: shell

   # Convert the ONNX model to top_mlir
   $ model_transform
   # Convert top_mlir to bmodel
   $ model_deploy


Website usage
------------------
After opening the webpage, select your final.mlir file.

.. figure:: ../assets/navigator.png
   :align: center

The page will automatically parse the final.mlir file and generate two visualizations: a peak memory utilization chart and a memory allocation Gantt chart.

.. figure:: ../assets/neuron_vis.png
   :align: center

The first chart shows peak memory usage over time on the device. Since memory management on the device resembles an overwrite operation—only tracking allocations but not explicit deallocations—this chart primarily helps identify at which timestep the peak memory usage occurs. Hovering your mouse over the chart displays the current memory allocation details.

The second chart provides a more precise and granular view of memory allocation status. The horizontal axis represents timesteps, and the vertical axis represents memory addresses. It visualizes the start/end timesteps and start/end addresses of each memory block. Hover tooltips are supported, and horizontal mouse-wheel zooming is enabled.

During analysis, users typically identify the peak timestep from the first chart, then zoom into that region in the second chart to locate the tensors and their corresponding loc identifiers occupying memory at that timestep. These can then be cross-referenced with the final.mlir file to analyze tensor lifetimes and assess whether memory consumption is reasonable. Tensor lifetimes are determined by the model’s topology, while memory footprint is jointly determined by tensor shape and data type.

Below is a zoomed-in view of the Gantt chart around the peak region shown above:

.. figure:: ../assets/neuron_detail.png
   :align: center
   :height: 8cm

Below is the corresponding final.mlir snippet for the peak region shown above:

.. code-block:: text

   %419 = "tpu.MatMul"(%415, %418, %0, %0, %0) {do_relu = false, dq_type = "NONE", fuse_rq = false, hdim_is_batch = false, input_zp = 0 : i64, is_lora = false, keep_dims = true, left_reuse = 1 : i64, left_transpose = false, multipliers = [1], output_transpose = false, q_group_size = 0 : i64, quant_mode = #tpu<rq_mode MultiplierShift>, relu_limit = -1.000000e+00 : f64, right_transpose = true, right_zp = 0 : i64, round_mode = #tpu<round_mode HalfAwayFromZero>, rshifts = [0]} : (tensor<1x4800x128xf32, 4788801536 : i64>, tensor<1x4800x128xf32, 4601942016 : i64>, none, none, none) -> tensor<1x4800x4800xf32, 4791259136 : i64> loc(#loc685)
   %420 = "tpu.MulConst"(%419) {const_val = 0.088388349161020605 : f64, do_relu = false, is_scalar = false, multiplier = 1 : si32, relu_limit = -1.000000e+00 : f64, rshift = 0 : si32} : (tensor<1x4800x4800xf32, 4791259136 : i64>) -> tensor<1x4800x4800xf32, 4601942016 : i64> loc(#loc686)
   %421 = "tpu.MulConst"(%411) {const_val = 0.088388349161020605 : f64, do_relu = false, is_scalar = false, multiplier = 1 : si32, relu_limit = -1.000000e+00 : f64, rshift = 0 : si32} : (tensor<1x60x80x60x80xf32, 4694102016 : i64>) -> tensor<1x60x80x60x80xf32, 4788801536 : i64> loc(#loc687)
   %422 = "tpu.Reshape"(%421) {flatten_start_dim = -1 : i64, shape = [1, 4800, 4800]} : (tensor<1x60x80x60x80xf32, 4788801536 : i64>) -> tensor<1x4800x4800xf32, 4788801536 : i64> loc(#loc688)
   %423 = "tpu.Softmax"(%422, %0, %0, %0, %0, %0) {axis = 2 : si32, beta = 1.000000e+00 : f64, log = false, round_mode = #tpu<round_mode HalfAwayFromZero>} : (tensor<1x4800x4800xf32, 4788801536 : i64>, none, none, none, none, none) -> tensor<1x4800x4800xf32, 4694102016 : i64> loc(#loc689)

As shown above, each line in mlir typically represents one operation (op). An op has input and output tensors, with the output tensor being defined on that line and subsequently used as an input to other operations in later lines. In addition to this information, the MLIR file includes a mapping table at the end that associates loc identifiers with original layer names(e.g., #loc686 = loc("/model/feature_flow_attn/Div_output_0_Div")). If users want to determine which part of the original model a given tensor corresponds to, they can look up the tensor's loc identifier in this table to find the corresponding layer name.

In the figure above, loc677 refers to a long-lived tensor defined earlier in the code and used as %411 to produce output %421 (i.e., loc687). Together, loc677, loc686, and loc687 (highlighted in blue) fully occupy memory during timesteps 301-302. Note that loc687 (the blue segment) overlaps spatially and temporally with the subsequent loc688. This occurs because reshape is an in-place operation—it reuses the same memory address for computation. Similar behaviors are also observed with operators like slice and concat.

In the following timesteps (303-304), memory is again fully occupied by loc688, loc689, and loc686. The extended lifetime of loc686 is due to its reuse as an input by later operations.

For more detailed analysis, you may examine the original ONNX model source to perform structural optimizations.



**Notes:**

Memory address allocation employs two strategies, and the better result from both is selected as the final allocation:

  * firstFit: Scans forward from the starting address to find the first available space that fits.
  * opSizeOrder: Sorts tensors by size (largest first) and allocates larger tensors first.

Under firstFit, input tensors are allocated contiguously at the beginning of the address space, whereas under opSizeOrder, they may be scattered across different addresses.

