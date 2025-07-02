.. _llm_convert:

编译LLM模型
===========================

概述
----

``llm_convert.py`` 是一个用于将大语言模型（LLM）转换成bmodel的工具，将原始模型权重转换为 bmodel 格式，以便在 BM1684X、BM1688 和 CV186AH 等芯片平台上进行高效推理。
目前支持的LLM类型包括qwen2/qwen3/qwen2.5vl/internvl3/llama等等。



命令行参数说明
----------------

下面介绍该工具支持的各个命令行参数：

- ``-m``, ``--model_path`` (string, 必填)
  指定原始模型权重的路径。例如：``./Qwen2-7B-Instruct``

- ``-s``, ``--seq_length`` (integer, 必填)
  指定转换时使用的序列长度。

- ``-q``, ``--quantize`` (string, 必填)
  指定 bmodel 的量化类型，必须从下面的选项中选择：

  - ``bf16``
  - ``w8bf16``
  - ``w4bf16``
  - ``f16``
  - ``w8f16``
  - ``w4f16``

- ``-g``, ``--q_group_size`` (integer, 默认值: 64)
  如果使用 W4A16 量化模式，则用于设置每组量化的组大小。

- ``-c``, ``--chip`` (string, 默认值: ``bm1684x``)
  指定生成 bmodel 的芯片平台，支持如下选项：

  - ``bm1684x``
  - ``bm1688``
  - ``cv186ah``

- ``--num_device`` (integer, 默认值: 1)
  指定 bmodel 部署的设备数。

- ``--num_core`` (integer, 默认值: 0)
  指定 bmodel 部署使用的核数，0表示采用最大核数。

- ``--symmetric``
  如果设置该标志，则使用对称量化。

- ``--max_input_length`` (integer, 默认值: 0)
  指定最大输入长度，0表示使用 ``seq_length`` 作为最大长度。

- ``--embedding_disk``
  如果设置该标志，则将word_embedding导出为二进制文件，并通过 CPU 进行推理。

- ``--max_pixels`` (integer)
  对于多模态模型如qwen2.5vl，用于指定最大像素尺寸，比如可以指定为 ``672,896``,表示 ``672x896`` 的图片；也可以是 ``602112``，表示最大像素。

- ``-o``, ``--out_dir`` (string, 默认值: ``./tmp``)
  指定输出的 bmodel 文件保存路径。

示例用法
---------

假设需要将位于 ``/workspace/Qwen2-7B-Instruct`` 的大模型转换为 ``bm1684x`` 平台的 bmodel，同时使用 ``384`` 的序列长度和``w4bf16`` 的量化类型，设置 group size 为 ``128``，并将输出文件存放在目录 ``qwen2_7b`` 下，可以执行以下命令：

首先需要将Qwen2-7B-Instruct从huggingface下载到本地，然后执行如下命令：

.. code-block:: bash

   llm_convert.py -m /workspace/Qwen2-7B-Instruct -s 384 -q w4bf16 -g 128 -c bm1684x -o qwen2_7b

注意如果提示transformers找不到，则需要安装，命令如下（其他pip包同理）：

.. code-block:: bash

   pip3 install transformers --upgrade

另外该命令也支持AWQ和GPTQ模型，参考命令如下：

.. code-block:: bash

   llm_convert.py -m /workspace/Qwen2.5-0.5B-Instruct-AWQ -s 384 -q w4bf16 -c bm1684x -o qwen2.5_0.5b
   llm_convert.py -m /workspace/Qwen2.5-0.5B-Instruct-GPTQ-Int4 -s 384 -q w4bf16 -c bm1684x -o qwen2.5_0.5b



