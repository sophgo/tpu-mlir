.. _llm_convert:

编译LLM模型
===========

概述
----

``llm_convert.py`` 用于将大语言模型（LLM，含多模态 LLM）的原始权重转换为
``.bmodel``，以便在芯片（``bm1684x``、``bm1688``、``cv186x``）上高效推理。

工具内部会完成：

- 通过 ``transformers.AutoConfig`` 读取模型配置，自动分发到对应的
  Converter（纯 LLM / 视觉 LLM / 语音 LLM / Omni）；
- 为每个 transformer block、embedding、LM head（以及多模态模型的
  vision tower）导出 MLIR；
- 编译并合并为单个 ``.bmodel``。

当前支持的模型族包括 ``qwen2``、``qwen3``、``qwen2_vl``、``qwen2_5_vl``、
``qwen3_vl``、``qwen3_5``、``llama``、``minicpm``、``minicpmv``、
``internvl_chat``、``chatglm``、``phi3``、``glm4v``、``gemma3``、
``mllama``、``paddleocr_vl``、``lfm2_vl``、``janus``、``qwen2_5_omni``、
``qwen3_asr``。
直接支持 AWQ / GPTQ / AutoRound 等已量化权重。

环境准备
--------

在安装 ``tpu_mlir`` whl 或克隆源码之后，先安装 Python 依赖：

.. code-block:: bash

   pip3 install -r requirements.txt

如果 ``AutoConfig.from_pretrained`` 在新模型上报错，请升级
``transformers`` 与 ``torchvision``：

.. code-block:: bash

   pip3 install -U transformers torchvision

命令行参数说明
--------------

必填参数
~~~~~~~~

- ``-m``, ``--model_path`` *(string)*
  原始模型目录，如 ``./Qwen3.5-2B-int4-AutoRound``。

- ``-s``, ``--seq_length`` *(int)*
  bmodel 支持的最大序列长度。

量化相关
~~~~~~~~

- ``-q``, ``--quantize`` *(默认: ``auto``)*
  可选 ``auto``、``bf16``、``f16``、``w8bf16``、``w4bf16``、``w8f16``、
  ``w4f16``。``auto`` 会优先沿用模型自带的量化配置（AWQ / GPTQ /
  AutoRound），否则退化为 ``bf16``。

- ``-g``, ``--q_group_size`` *(int, 默认: ``64``)*
  W4 per-group 量化的组大小。

- ``--symmetric``
  使用对称量化。

目标硬件
~~~~~~~~

- ``-c``, ``--chip``, ``--processor`` *(默认: ``bm1684x``)*
  目标芯片：``bm1684x``、``bm1688``、``cv186x``。

- ``--num_device`` *(int, 默认: ``1``)*
  bmodel 部署的设备数。

- ``--distribute_strategy`` *(``tp`` | ``pp``, 默认: ``tp``)*
  多卡并行策略，仅在 ``--num_device > 1`` 时生效。

- ``--num_core`` *(int, 默认: ``0``)*
  使用的核数，``0`` 表示使用芯片最大核数。

序列与 Prefill
~~~~~~~~~~~~~~

- ``-b``, ``--batch`` *(int, 默认: ``1``)*
- ``--max_input_length`` *(int, 默认: ``0`` ⇒ 等于 ``seq_length``)*
- ``--max_prefill_kv_length`` *(int, 默认: ``0`` ⇒ 等于 ``seq_length``)*
- ``--use_block_with_kv``
  Prefill 阶段复用历史 KV。``--share_prompt`` 会自动开启。
- ``--share_prompt``
  多轮对话共享同一段 prompt 前缀。

视觉相关（仅多模态模型）
~~~~~~~~~~~~~~~~~~~~~~~~

- ``--max_pixels`` *(``W,H``)*
  Vision tower 支持的最大输入尺寸，格式为 ``宽,高``（如 ``672,896``）。
  未指定时按 ``model_type`` 选择默认：

  ===============  ===============
  ``model_type``   默认值
  ===============  ===============
  ``qwen2_5_vl``   ``672,896``
  ``minicpmv``     ``980,980``
  其它             ``768,768``
  ===============  ===============

  对于 Qwen2-VL / Qwen2.5-VL / GLM4V / MiniCPM-V，``W*H`` 必须是
  ``28*28`` 的整数倍；对于 Qwen3-VL / Qwen3.5，必须是 ``32*32`` 的整数
  倍。工具会在编译前校验。

LoRA / 采样 / 其他
~~~~~~~~~~~~~~~~~~

- ``--lora_max_rank`` *(int, 默认: ``0``)*
  预留 LoRA 容量；``0`` 表示不开启 LoRA。
- ``--do_sample``
  在 LM head 之外额外导出采样 head（greedy + topk/topp）。
- ``--embedding_disk``
  将 word_embedding 导出为 ``.bin`` 并在 CPU 上推理，节省片上内存。
- ``--dynamic``
  Prefill 启用动态 shape。``qwen3_5`` 始终启用。
- ``--rvti``
  启用 RVTI；仅适用于 ``bm1690e``。

流程控制
~~~~~~~~

- ``-o``, ``--out_dir`` *(string)*
  输出目录。未指定时默认 ``./<model_name>_<chip>_<quantize>`` 并打印到
  终端。
- ``--only_mlir``
  导出 MLIR 后即停止，不编译为 bmodel。
- ``--again``
  续跑：跳过 ``--out_dir`` 中已存在产物的阶段。
- ``--debug``
  保留中间临时文件，方便调试。
- ``--dry_run``
  解析全部配置（默认值、模型相关默认、整除性检查、所选 Converter）并
  打印，不进行编译。建议在长时间转换前先跑一次。
- ``-V``, ``--version``
  打印底层 ``pymlir`` 版本。

示例
----

常规 LLM（Qwen2-7B-Instruct，``w4bf16``）::

   llm_convert.py -m /workspace/Qwen2-7B-Instruct \
       -s 384 -q w4bf16 -g 128 -c bm1684x \
       -o qwen2_7b

已量化模型（AWQ / GPTQ 用法相同）::

   llm_convert.py -m /workspace/Qwen2.5-0.5B-Instruct-AWQ        -s 384 -c bm1684x -o qwen2.5_0.5b
   llm_convert.py -m /workspace/Qwen2.5-0.5B-Instruct-GPTQ-Int4  -s 384 -c bm1684x -o qwen2.5_0.5b

Qwen3.5 多模态（AutoRound int4）::

   llm_convert.py -m /workspace/Qwen3.5-2B-int4-AutoRound \
       --max_input_length 1024 -s 2048 -c bm1684x \
       --max_pixels 768,768 -o qwen3.5_2b

启动长时间转换前，先用 ``--dry_run`` 确认配置::

   llm_convert.py -m /workspace/Qwen3.5-2B-int4-AutoRound \
       -s 2048 -c bm1684x --dry_run

dry-run 会打印解析后的芯片、量化、序列长度、``max_shape`` /
``max_pixels`` 以及最终选中的 Converter 类，便于在数小时编译之前发现
拼写错误或参数冲突。
