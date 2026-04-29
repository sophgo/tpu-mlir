.. _llm_convert:

Compile LLM Model
=================

Overview
--------

``llm_convert.py`` converts the original weights of a Large Language Model
(LLM) — including multimodal LLMs — into a ``.bmodel`` for efficient
inference on chips (``bm1684x``, ``bm1688``, ``cv186x``).

The tool takes care of:

- loading the model config with ``transformers.AutoConfig`` and
  dispatching to the appropriate converter (LLM / VLM / ASR / OMNI);
- exporting MLIR for each transformer block, the embedding, the LM head,
  and (for VLMs) the vision tower;
- compiling and combining everything into a single ``.bmodel``.

Supported model families include ``qwen2``, ``qwen3``, ``qwen2_vl``,
``qwen2_5_vl``, ``qwen3_vl``, ``qwen3_5``, ``llama``, ``minicpm``,
``minicpmv``, ``internvl_chat``, ``chatglm``, ``phi3``, ``glm4v``,
``gemma3``, ``mllama``, ``paddleocr_vl``, ``lfm2_vl``, ``janus``,
``qwen2_5_omni``, ``qwen3_asr``.
AWQ / GPTQ / AutoRound pre-quantized weights are accepted directly.

Prerequisites
-------------

Install Python dependencies once after installing the ``tpu_mlir`` wheel
or after cloning the source tree:

.. code-block:: bash

   pip3 install -r requirements.txt

If ``AutoConfig.from_pretrained`` fails for a recent model, upgrade
``transformers`` and ``torchvision``:

.. code-block:: bash

   pip3 install -U transformers torchvision

Command-Line Arguments
----------------------

Required
~~~~~~~~

- ``-m``, ``--model_path`` *(string)*
  Path to the original model directory, e.g. ``./Qwen3.5-2B-int4-AutoRound``.

- ``-s``, ``--seq_length`` *(int)*
  Maximum sequence length supported by the compiled bmodel.

Quantization
~~~~~~~~~~~~

- ``-q``, ``--quantize`` *(default: ``auto``)*
  One of ``auto``, ``bf16``, ``f16``, ``w8bf16``, ``w4bf16``, ``w8f16``,
  ``w4f16``. ``auto`` honors a pre-quantized config when present
  (AWQ / GPTQ / AutoRound), otherwise falls back to ``bf16``.

- ``-g``, ``--q_group_size`` *(int, default: ``64``)*
  Group size for per-group W4 quantization.

- ``--symmetric``
  Use symmetric quantization.

Target hardware
~~~~~~~~~~~~~~~

- ``-c``, ``--chip``, ``--processor`` *(default: ``bm1684x``)*
  Target chip: ``bm1684x``, ``bm1688``, ``cv186x``.

- ``--num_device`` *(int, default: ``1``)*
  Number of devices the bmodel will run on.

- ``--distribute_strategy`` *(``tp`` | ``pp``, default: ``tp``)*
  Multi-device strategy. Only takes effect when ``--num_device > 1``.

- ``--num_core`` *(int, default: ``0``)*
  Number of cores. ``0`` means use the chip's maximum.

Sequence and prefill
~~~~~~~~~~~~~~~~~~~~

- ``-b``, ``--batch`` *(int, default: ``1``)*
- ``--max_input_length`` *(int, default: ``0`` ⇒ same as ``seq_length``)*
- ``--max_prefill_kv_length`` *(int, default: ``0`` ⇒ same as ``seq_length``)*
- ``--use_block_with_kv``
  Reuse history KV during prefill. Required for ``--share_prompt``.
- ``--share_prompt``
  Share the same prompt prefix across multiple dialogs.

Vision (multimodal models only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``--max_pixels`` *(``W,H``)*
  Maximum image size for the vision tower, given as
  ``width,height`` (e.g. ``672,896``). When omitted, a default is
  picked by model type:

  ===============  ===============
  ``model_type``   default
  ===============  ===============
  ``qwen2_5_vl``   ``672,896``
  ``minicpmv``     ``980,980``
  others           ``768,768``
  ===============  ===============

  ``W*H`` must be a multiple of ``28*28`` for Qwen2-VL / Qwen2.5-VL /
  GLM4V / MiniCPM-V, and a multiple of ``32*32`` for Qwen3-VL / Qwen3.5.
  The tool validates this up front.

LoRA / sampling / misc
~~~~~~~~~~~~~~~~~~~~~~

- ``--lora_max_rank`` *(int, default: ``0``)*
  Reserve LoRA capacity. ``0`` disables LoRA support.
- ``--do_sample``
  Add a sampling head separate from the LM head (greedy + topk/topp).
- ``--embedding_disk``
  Export the word-embedding as a ``.bin`` file and run it on the CPU
  (saves on-chip memory).
- ``--dynamic``
  Enable dynamic shape compilation for prefill. ``qwen3_5`` always
  enables this.

Workflow control
~~~~~~~~~~~~~~~~

- ``-o``, ``--out_dir`` *(string)*
  Output directory. If omitted, defaults to
  ``./<model_name>_<chip>_<quantize>`` and is printed to stdout.
- ``--only_mlir``
  Stop after MLIR export; do not compile to bmodel.
- ``--again``
  Resume an interrupted run by skipping stages whose outputs already
  exist in ``--out_dir``.
- ``--debug``
  Keep intermediate temp files for debugging.
- ``--dry_run``
  Resolve the configuration (defaults, per-model overrides, divisibility
  checks, chosen converter class) and print it without compiling.
  Recommended before launching a long conversion.
- ``-V``, ``--version``
  Print the underlying ``pymlir`` version.

Examples
--------

Plain LLM (Qwen2-7B-Instruct, ``w4bf16``)::

   llm_convert.py -m /workspace/Qwen2-7B-Instruct \
       -s 384 -q w4bf16 -g 128 -c bm1684x \
       -o qwen2_7b

Pre-quantized weights (AWQ / GPTQ work the same way)::

   llm_convert.py -m /workspace/Qwen2.5-0.5B-Instruct-AWQ        -s 384 -c bm1684x -o qwen2.5_0.5b
   llm_convert.py -m /workspace/Qwen2.5-0.5B-Instruct-GPTQ-Int4  -s 384 -c bm1684x -o qwen2.5_0.5b

Qwen3.5 multimodal (AutoRound int4)::

   llm_convert.py -m /workspace/Qwen3.5-2B-int4-AutoRound \
       --max_input_length 1024 -s 2048 -c bm1684x \
       --max_pixels 768,768 -o qwen3.5_2b

Sanity-check first with ``--dry_run``::

   llm_convert.py -m /workspace/Qwen3.5-2B-int4-AutoRound \
       -s 2048 -c bm1684x --dry_run

The dry-run prints the resolved chip, quantization, sequence lengths,
``max_shape`` / ``max_pixels``, and the converter class that will be
used — useful for catching typos before a multi-hour compile.
