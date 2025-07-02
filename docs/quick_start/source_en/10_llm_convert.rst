.. _llm_convert:

Compile LLM Model
===========================

Overview
--------

``llm_convert.py`` is a tool for converting large language models (LLM) into the bmodel format. It converts the original model weights to the bmodel format, enabling efficient inference on chip platforms such as BM1684X, BM1688, and CV186AH.
Currently supported LLM types include qwen3/qwen2/qwen2.5vl/internvl3/llama, etc.

Command-Line Arguments
------------------------

Below is an explanation of the command-line arguments supported by this tool:

- ``-m``, ``--model_path`` (string, required)
  Specify the path to the original model weights.
  For example: ``./Qwen2-7B-Instruct``

- ``-s``, ``--seq_length`` (integer, required)
  Specify the sequence length to be used during the conversion.

- ``-q``, ``--quantize`` (string, required)
  Specify the quantization type for the bmodel. You must choose from the following options:

  - ``bf16``
  - ``w8bf16``
  - ``w4bf16``
  - ``f16``
  - ``w8f16``
  - ``w4f16``

- ``-g``, ``--q_group_size`` (integer, default: 64)
  When using the W4A16 quantization mode, this sets the group size for quantization.

- ``-c``, ``--chip`` (string, default: ``bm1684x``)
  Specify the chip platform for generating the bmodel. Supported options are:

  - ``bm1684x``
  - ``bm1688``
  - ``cv186ah``

- ``--num_device`` (integer, default: 1)
  Specify the number of devices for bmodel deployment.

- ``--num_core`` (integer, default: 0)
  Specify the number of cores to be used for bmodel deployment, where 0 indicates using the maximum number of cores.

- ``--symmetric``
  Set this flag to use symmetric quantization.

- ``--max_input_length`` (integer, default: 0)
  Specify the maximum input length. A value of 0 means using ``seq_length`` as the maximum length.

- ``--embedding_disk``
  Set this flag to export the word_embedding as a binary file and run inference on the CPU.

- ``--max_pixels`` (integer)
  For multimodal models such as qwen2.5vl, it is used to specify the maximum pixel dimensions. For example, you can specify 672,896, indicating an image of 672x896; or it can be 602112, representing the maximum number of pixels.

- ``-o``, ``--out_dir`` (string, default: ``./tmp``)
  Specify the output directory for the generated bmodel files.

Example Usage
--------------

Assume you need to convert a large model located at ``/workspace/Qwen2-7B-Instruct`` into a bmodel for the ``bm1684x`` platform, using a sequence length of ``384`` and the ``w4bf16`` quantization type. Additionally, set the group size to ``128`` and store the output files in the directory ``qwen2_7b``. You can execute the following command:

First, download Qwen2-7B-Instruct locally from Hugging Face, then run:

.. code-block:: bash

   llm_convert.py -m /workspace/Qwen2-7B-Instruct -s 384 -q w4bf16 -g 128 -c bm1684x -o qwen2_7b

Note: If you encounter an error indicating that transformers is not found, you will need to install it. The command is as follows (apply a similar approach for other pip packages):

.. code-block:: bash

   pip3 install transformers --upgrade

Also supports AWQ and GPTQ models, such as Qwen2.5-0.5B-Instruct-AWQ and Qwen2.5-0.5B-Instruct-GPTQ-Int4. The conversion command is as follows:

.. code-block:: bash

   llm_convert.py -m /workspace/Qwen2.5-0.5B-Instruct-AWQ -s 384 -q w4bf16 -c bm1684x -o qwen2.5_0.5b
   llm_convert.py -m /workspace/Qwen2.5-0.5B-Instruct-GPTQ-Int4 -s 384 -q w4bf16 -c bm1684x -o qwen2.5_0.5b

