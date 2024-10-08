前端转换
============

本章以 onnx 模型为例介绍模型/算子在本工程中的前端转换流程。

主要工作
----------------
前端主要负责将原始模型转换为 Top 层(硬件无关层)mlir 模型的工作(不包含 Canonicalize 部分, 因
此生成的文件名为“\*_origin.mlir”), 这个过程会根据原始模型与运行 model_transform.py 时输入的参数逐
一创建并添加对应的算子(Op), 最终生成 mlir 文件与保存权重的 npz 文件。

工作流程
----------------
1. 前提(Prereq): Top 层算子定义, 该部分内容保存在 TopOps.td 文件

2. 输入(Input): 输入原始 onnx 模型与参数(主要是预处理参数)

3. 初始化 OnnxConverter(load_onnx_model + initMLIRImporter)

    * load_onnx_model 部分主要是对模型进行精简化, 根据 arguments 中的 output_names 截取模型, 并提取精简后模型的相关信息

    * init_MLIRImporter 部分主要是生成初始的 mlir 文本

4. generate_mlir

    * 依次创建 input op, 模型中间 nodes op 以及 return op, 并将其补充到 mlir 文本中(如果该op带有权重, 则会额外创建weight op)

5. 输出(Output)

    * 将精简后的模型保存为“\*_opt.onnx”文件

    * 生成”.prototxt”文件保存除权重外的模型信息

    * 将生成的文本转为字符串并保存为“.mlir”文件

    * 将模型权重(tensors)保存为“.npz”文件


前端转换的工作流程如图所示(:ref:`mlir_convert`)。

.. _mlir_convert:
.. figure:: ../assets/mlir_convert.png
   :align: center

   前端转换流程


补充说明:
  * Build input op 需要:

     1. input_names

     2. 每个 input 对应的 index

     3. 预处理参数(若输入为图像)

  * Convert nodes op 需要:

     1. 从 operands 获取该 node 的输入 op(即前一个已经 build 或 convert 好的算子)

     2. 从 shapes 中获取 output_shape

     3. 从 onnx node 中提取的 attrs。Attrs 会通过 MLIRImporter 设定为与 TopOps.td 定义一一对应的属性

  * Build return op 需要:

      依照 output_names 从 operands 获取相应的 op

  * 每创建或者转换一个算子都会执行一次插入操作, 将算子插入到 mlir 文本中, 使最终生成的文本能从头到尾与原 onnx 模型一一对应


算子转换样例
----------------

本节以 Conv 算子为例, 将单 Conv 算子的 onnx 模型转换为 Top mlir, 原模型如图所示(:ref:`conv_op`)

.. _conv_op:
.. figure:: ../assets/conv_op.png
   :align: center
   :height: 15cm

   单 Conv 模型


转换流程为:

1. 算子定义

  在 TopOps.td 中定义 Top.Conv 算子, 算子定义如图所示(:ref:`convop_def`)

.. _convop_def:
.. figure:: ../assets/convop_def.png
   :align: center
   :height: 15cm

   Conv 算子定义


2. 初始化 OnnxConverter

  load_onnx_model:

  * 由于本例使用的是最简模型, 所以生成的 Conv_opt.onnx 模型与原模型相同。

  * input_names 保存了 Conv 算子的输入名“input”

  * tensors 中保存了 Conv 算子的权重 weight 与 bias

  * shapes 中保存了Conv算子的输入和输出shape。

  * output_names 中保存了 Conv 算子的输出名“output”

  init_MLIRImporter:

  根据 input_names 与 output_names 从 shapes 中获取了对应的 input_shape 与 output_shape, 加上model_name, 生成了初始的 mlir 文本 MLIRImporter.mlir_module, 如图所示(:ref:`origin_mlir`)。

.. _origin_mlir:
.. figure:: ../assets/origin_mlir.png
   :align: center

   初始 mlir 文本


3. generate_mlir

   * build input op, 生成的 Top.inputOp 会被插入到 MLIRImporter.mlir_module 中。

   * 根据 node.op_type (即“ Conv ”) 调用 convert_conv_op() ,  该函数中会调用MLIRImporter.create_conv_op 来创建 ConvOp, 而 create 函数需要的参数有:

      1) 输入 op: 从(:ref:`conv_op`)可知, Conv 算子的 inputs 一共包含了 input, weight 与 bias, inputOp 已被创建好, weight 与 bias 的 op 则通过 getWeightOp()创建。

      2) output_shape: 利用 onnx_node.name 从 shapes 中获取 Conv 算子的输出shape。

      3) Attributes: 从 onnx Conv 算子中获取如(:ref:`conv_op`)中的 attributes。

         在 create 函数里 Top.Conv 算子的 attributes 会根据(:ref:`convop_def`)中的定义来设定。Top.ConvOp 创建后会被插入到 mlir 文本中

   * 根据 output_names 从 operands 中获取相应的 op, 创建 return_op 并插入到 mlir 文本中。到此为止, 生成的 mlir 文本如图所示(:ref:`mlir_txt`)。

.. _mlir_txt:
.. figure:: ../assets/mlir_txt.png
   :align: center

   完整的 mlir 文本


4. 输出

  将 mlir 文本保存为 Conv_origin.mlir, tensors 中的权重保存为 Conv_TOP_F32_all_weight.npz。

