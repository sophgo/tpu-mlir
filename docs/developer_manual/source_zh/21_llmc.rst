LLMC使用指南
========================

TPU-MLIR weight-only 量化
--------------------------

TPU-MLIR中支持对大模型进行weight-only仅权重量化,采用的量化算法是RTN(round to nearest)算法,量化粒度为per-channel或者per-group。具体的量化配置如下:

.. list-table:: weight-only量化参数
   :widths: 25 25 25 25
   :header-rows: 1

   * - bit
     - symmetric
     - granularity
     - group_size
   * - 4
     - False
     - per-channel or per-group
     - -1 or 64(default)
   * - 8
     - True
     - per-channel
     - -1

RTN量化算法简洁高效,但也面临一些不足,对于一些要求模型精度比较高的场景,RTN算法量化的模型可能无法满足精度需求,此时需要借助大模型量化工具llmc_tpu来进一步提升精度。

llmc_tpu
-------------------------------

本项目源自 `ModelTC/llmc <https://github.com/ModelTC/llmc>`_。ModelTC/llmc是非常优秀的项目,专为压缩LLM设计,利用最先进的压缩算法提高效率并减少模型体积,同时不影响预测精度。如果要深入了解llmc项目,请转到 `<https://github.com/ModelTC/llmc>`_。

本项目是基于 `ModelTC/llmc` 进行一些定制化修改,用于支持Sophgo处理器。

环境准备
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **下载本项目**

.. code-block:: shell
   :linenos:

   git clone git@github.com:sophgo/llmc-tpu.git


2. **准备您需要量化的LLM或者VLM模型,放到`llmc-tpu`的同级目录**

比如huggingface上下载 `Qwen2-VL-2B-Instruct`，如下：

.. code-block:: shell
   :linenos:

   git lfs install
   git clone git@hf.co:Qwen/Qwen2-VL-2B-Instruct


3. **下载Docker并建立Docker容器**

pull docker images

.. code-block:: shell
   :linenos:

   docker pull registry.cn-hangzhou.aliyuncs.com/yongyang/llmcompression:pure-latest

create container. llmc_test is just a name, and you can set your own name

.. code-block:: shell
   :linenos:

   docker run --privileged --name llmc_test -it --shm-size 64G --gpus all -v $PWD:/workspace  registry.cn-hangzhou.aliyuncs.com/yongyang/llmcompression:pure-latest


4. **进入`llmc-tpu`，安装依赖包**

注意现在已经在docker容器中

.. code-block:: shell
   :linenos:

   cd /workspace/llmc-tpu
   pip3 install -r requirements.txt

tpu 目录
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

├── README.md
├── data
│   ├──LLM
│      ├──cali                              #校准数据集
│      ├──eval                              #推理数据集
│   ├──VLM
│      ├──cali
│      ├──eval
├── config
│   ├──LLM                                  #LLM量化config
│      ├── Awq.yml                              #Awq config
│      ├── GPTQ.yml                             #GPTQ config
│   ├──VLM                                  #VLM量化config
│      ├── Awq.yml                              #Awq config
├── example.yml                             #量化参数参考例子
├── llm_quant.py                            #量化主程序
├── run_llmc.sh                             #量化运行脚本

操作步骤
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【阶段一】准备校准数据集和测试数据集
""""""""""""""""""""""""""""""""""""""""

* 注意点1: **校准数据集** 可以是开源数据集或者业务数据集，如果模型经过下游业务数据集微调，则需要选用业务数据集做校准
* 注意点2: **测试数据集** 主要用来评估当前模型的精度表现,包括预训练(pretrain)模型或者量化(fake_quant)模型的精度

可以选择用开源数据集，也可以选择用业务数据集。

开源数据集
~~~~~~~~~~~~~~~~~~~~~~~

如果有业务数据集最好，没有的话可以用开源数据集，如下：

.. list-table:: 数据集选取
   :widths: 25 25 25 25
   :header-rows: 1

   * - 模型类型
     - 量化算法
     - 校准数据集(开源)
     - 测试数据集(开源)
   * - LLM
     - Awq
     - pileval
     - wikitext2
   * - LLM
     - GPTQ
     - wikitext2
     - wikitext2
   * - VLM
     - Awq
     - MME
     - MME

校准数据集的选取与模型类型和量化算法相关,例如如果量化的是LLM模型,使用的是Awq算法,通常推荐使用pileval数据集作为校准集。针对这些开源数据集本文档提供了对应的下载命令,可以运行下载相应的数据集。具体操作如下:可打开llmc-tpu/tools文件,里面对应有download_calib_dataset.py和download_eval_dataset.py两个python脚本,分别用于下载校准集和测试集。

如果是VLM模型,建议使用Awq算法,下载数据集命令如下:

.. code-block:: shell
   :linenos:

   cd /workspace/llmc-tpu

* 校准数据集

.. code-block:: shell
   :linenos:

   python3 tools/download_calib_dataset.py --dataset_name MME --save_path tpu/data/VLM/cali

* 测试数据集

.. code-block:: shell
   :linenos:

   python3 tools/download_eval_dataset.py --dataset_name MME --save_path tpu/data/VLM/eval


如果是LLM模型,建议用Awq算法,下载数据集命令如下:

.. code-block:: shell
   :linenos:

   cd /workspace/llmc-tpu

* 校准数据集

.. code-block:: shell
   :linenos:

   python3 tools/download_calib_dataset.py --dataset_name pileval --save_path tpu/data/LLM/cali

* 测试数据集

.. code-block:: shell
   :linenos:

   python3 tools/download_eval_dataset.py --dataset_name wikitext2 --save_path tpu/data/LLM/eval

业务数据集
~~~~~~~~~~~~~

1. **业务校准数据集**

如果模型经过下游业务数据集微调，在选择校准集时，通常应该选择业务数据集。
* 如果是LLM,将业务数据集放置于上述LLM/cali目录下即可。至于数据集具体的格式,用户可以将一条一条数据文本,写到txt文件里面,每一行代表一条文本数据，使用上述的配置，可以实现自定义数据集的校准。
* 如果是VLM,将业务数据集放置于上述VLM/cali目录下即可。至于数据集具体的格式,可以参考VLM/cali/general_custom_data中的格式,选择符合需求的格式即可。这里一定需要注意,最后的json文件应该命名为samples.json。

2. **业务测试数据集**

如果模型经过下游业务数据集校准，在选择测试集时，通常应该选择业务数据集测试。
* 如果是LLM,将业务数据集放置于上述LLM/eval目录下即可。至于数据集具体的格式,用户可以将一条一条数据文本,写到txt文件里面,每一行代表一条文本数据，使用上述的配置，可以实现自定义数据集的测试。
* 如果是VLM,将业务数据集放置于上述VLM/eval目录下即可。至于数据集具体的格式,可以参考VLM/cali/general_custom_data中的格式,选择符合需求的格式即可。这里一定需要注意,最后的json文件应该命名为samples.json。


【阶段二】配置量化config文件
""""""""""""""""""""""""""""""""

* 注意点:量化config文件包括了量化过程中所需的量化配置,用户可按照需求进行选择,同时为了对齐TPU硬件的配置也会对某些参数做出限制,具体可看下文详细介绍。

config文件参数说明
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml
   :linenos:

    base:
        seed: &seed 42
    model:
        type: Qwen2VL # 设置模型名, 具体支持的模型参见llmc/models目录
        path: /workspace/Qwen2-VL-2B-Instruct    # 设置模型权重路径，请改成您需要的模型
        torch_dtype: auto
    calib:
        name: mme   # 设置成实际的校准数据集名称，mme，pileval等等
        download: False
        path: /workspace/llmc-tpu/tpu/data/VLM/cali/MME  # 设置校准数据集路径
        n_samples: 128
        bs: 1
        seq_len: 512
        preproc: pileval_awq
        seed: *seed
    eval:
        eval_pos: [pretrain, fake_quant]
        name: mme  # 设置成实际的测试数据集名称，mme,wikitext2等等
        download: False
        path: /workspace/llmc-tpu/tpu/data/VLM/eval/MME # 设置测试数据集路径
        bs: 1
        seq_len: 2048
    quant:
        method: Awq
        quant_objects: [language] # 默认只量化LLM部分，如要量化VIT部分，则设置成[vision, language]
        weight:
            bit: 4 # 设置成想要的量化bit，可以支持4或8
            symmetric: False # 4bit填False；8bit填True
            granularity: per_group # 4bit填per_group；8bit，填per_channel
            group_size: 64 # 4bit填64(与TPU-MLIR对应)；8bit, 填-1
        special:
            trans: True
            trans_version: v2
            weight_clip: True
            clip_sym: True
    save:
        save_trans: True       # 当设置为True，可以保存下调整之后的浮点权重
        save_path: ./save_path # 设置保存权重的路径
    run:
        task_name: awq_w_only
        task_type: VLM   # 设置成VLM或者LLM


上面是以Awq算法为例构建的一个完整的config文件。为了简便用户操作,用户可以将上面直接拷贝到自己的config中,然后对有注解的部分参数进行修改。
下面对重要的一些参数做详细的说明：

.. list-table:: 相关参数介绍
   :widths: 25 60
   :header-rows: 1

   * - 参数
     - 描述
   * - model
     - 模型名称,支持的模型在llmc/models目录,可以自行支持新模型 `llmc/models/xxxx.py`
   * - calib
     - calib类参数主要指定校准集相关的参数
   * - eval
     - eval类参数主要指定了和测试集相关的参数
   * - quant
     - 指定量化参数,一般建议用Awq算法,quant_objects一般选language,关于weight量化参数参考下表

为了与`TPU-MLIR`对齐,weight量化相关参数配置如下:

.. list-table:: weight-only量化参数
   :widths: 25 25 25 25
   :header-rows: 1

   * - bit
     - symmetric
     - granularity
     - group_size
   * - 4
     - False
     - per-channel or per-group
     - -1 or 64(default)
   * - 8
     - True
     - per-channel
     - -1

【阶段三】执行量化算法
""""""""""""""""""""""""""""""""

.. code-block:: shell
   :linenos:

   cd /workspace/llmc-tpu
   python3 tpu/llm_quant.py --llmc_tpu_path . --config_path ./tpu/example.yml

* config_path则表示量化config文件对应的路径,llmc_tpu_path表示当前llmc_tpu路径
