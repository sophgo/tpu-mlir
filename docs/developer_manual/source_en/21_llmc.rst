LLMC Guidance
========================================

TPU-MLIR weight-only quantization
-------------------------------------------------------

TPU-MLIR supports weight-only quantization for large models, utilizing the RTN (round to nearest) quantization algorithm with a quantization granularity of per-channel or per-group. The specific quantization configurations are as follows:

.. list-table:: weight-only quantization parameters
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

The RTN quantization algorithm is straightforward and efficient, but it also has some limitations. In scenarios that require higher model accuracy, models quantized using the RTN algorithm may not meet the precision requirements. In such cases, it is necessary to utilize the large model quantization tool llmc_tpu to further enhance accuracy.

llmc_tpu
-------------------------------

This project originates from `ModelTC/llmc <https://github.com/ModelTC/llmc>`_. `ModelTC/llmc` is an excellent project specifically designed for compressing Large Language Models (LLMs). It leverages state-of-the-art compression algorithms to enhance efficiency and reduce model size without compromising prediction accuracy. If you want to learn more about the llmc project, please visit `<https://github.com/ModelTC/llmc>`_.

This project is based on `ModelTC/llmc` with some customized modifications to support the Sophgo processor.

Environment Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Download This Project**

.. code-block:: shell
   :linenos:

   git clone git@github.com:sophgo/llmc-tpu.git


2. **Prepare the LLM or VLM Model for Quantization,Place the model you need to quantize in the same-level directory as llmc-tpu**

For Example: Download `Qwen2-VL-2B-Instruct` from Huggingface

.. code-block:: shell
   :linenos:

   git lfs install
   git clone git@hf.co:Qwen/Qwen2-VL-2B-Instruct


3. **Download Docker and Set Up a Docker Container**

pull docker images

.. code-block:: shell
   :linenos:

   docker pull registry.cn-hangzhou.aliyuncs.com/yongyang/llmcompression:pure-latest

create container. llmc_test is just a name, and you can set your own name

.. code-block:: shell
   :linenos:

   docker run --privileged --name llmc_test -it --shm-size 64G --gpus all -v $PWD:/workspace  registry.cn-hangzhou.aliyuncs.com/yongyang/llmcompression:pure-latest


4. **Enter llmc-tpu Directory and Install Dependencies**

Note that you are already in a Docker container.

.. code-block:: shell
   :linenos:

   cd /workspace/llmc-tpu
   pip3 install -r requirements.txt

tpu Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

├── README.md
├── data
│   ├──LLM
│      ├──cali                              #Calibration Dataset
│      ├──eval                              #Eval Dataset
│   ├──VLM
│      ├──cali
│      ├──eval
├── config
│   ├──LLM                                  #LLM quant config
│      ├── Awq.yml                              #Awq config
│      ├── GPTQ.yml                             #GPTQ config
│   ├──VLM                                  #VLM quant config
│      ├── Awq.yml                              #Awq config
├── example.yml                             #Quantization Parameters Reference Example
├── llm_quant.py                            #Quantization Main Program
├── run_llmc.sh                             #Quantization Run Script

Operating Steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[Phase 1] Prepare Calibration and Eval Datasets
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

* Note 1: **Calibration Dataset** can be an open-source dataset or a business dataset. If the model has been fine-tuned on downstream business datasets, then a business dataset needs to be selected for calibration.
* Note 2: **Eval Dataset** is primarily used to evaluate the accuracy performance of the current model, including the accuracy of pre-trained (pretrain) models or quantized (fake_quant) models.

You can choose to use an open-source dataset or a business dataset.

open-source dataset
~~~~~~~~~~~~~~~~~~~~~~~

If a business dataset is available, it is preferable. If not, you can use an open-source dataset as follows:

.. list-table:: Dataset Selection
   :widths: 25 25 25 25
   :header-rows: 1

   * - Model Type
     - Quantization Algorithm
     - Calibration Dataset (Open-source)
     - Eval Dataset (Open-source)
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

The selection of the calibration dataset depends on the model type and quantization algorithm. For example, if the model being quantized is an LLM and uses the Awq algorithm, it is typically recommended to use the Pileval dataset as the calibration set. For these open-source datasets, this document provides the corresponding download commands, which can be executed to download the respective datasets. The specific steps are as follows: open the llmc-tpu/tools directory, where you will find two Python scripts, download_calib_dataset.py and download_eval_dataset.py, which are used to download the calibration and eval datasets, respectively.

If it is a VLM model, it is recommended to use the Awq algorithm. The command to download the dataset is as follows:

.. code-block:: shell
   :linenos:

   cd /workspace/llmc-tpu

* Calibration Dataset

.. code-block:: shell
   :linenos:

   python3 tools/download_calib_dataset.py --dataset_name MME --save_path tpu/data/VLM/cali

* Eval Dataset

.. code-block:: shell
   :linenos:

   python3 tools/download_eval_dataset.py --dataset_name MME --save_path tpu/data/VLM/eval


If it is an LLM model, it is recommended to use the Awq algorithm. The command to download the dataset is as follows:

.. code-block:: shell
   :linenos:

   cd /workspace/llmc-tpu

* Calibration Dataset

.. code-block:: shell
   :linenos:

   python3 tools/download_calib_dataset.py --dataset_name pileval --save_path tpu/data/LLM/cali

* Eval Dataset

.. code-block:: shell
   :linenos:

   python3 tools/download_eval_dataset.py --dataset_name wikitext2 --save_path tpu/data/LLM/eval

business dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **business calibration dataset**

If the model has been fine-tuned on downstream business datasets, it is generally recommended to select the business dataset when choosing the calibration set.
* If it is an LLM, simply place the business dataset in the aforementioned LLM/cali directory. Regarding the specific format of the dataset, users can write each data entry as separate lines in a .txt file, with each line representing a single text data entry. By using the above configuration, you can perform calibration with a custom dataset.
* If it is a VLM, simply place the business dataset in the aforementioned VLM/cali directory. Regarding the specific format of the dataset, you can refer to the format in VLM/cali/general_custom_data and choose the format that meets your needs. It is important to note that the final JSON file should be named samples.json.

2. **business eval dataset**

If the model has been calibrated with downstream business datasets, it is generally recommended to use a business dataset for eval when selecting the eval set.
* If it is an LLM, simply place the business dataset in the aforementioned LLM/eval directory. Regarding the specific format of the dataset, users can write each data entry as a separate line of text in a .txt file, with each line representing one text data entry. Using the above configuration, custom dataset testing can be achieved.
* If it is a VLM, simply place the business dataset in the aforementioned VLM/eval directory. Regarding the specific format of the dataset, you can refer to the format in VLM/cali/general_custom_data and choose the format that meets your needs. It is important to note that the final JSON file should be named samples.json.


Phase Two: Configure the Quantization Configuration File
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

* Note: The quantization configuration file includes the settings required for the quantization process. Users can select configurations according to their needs. Additionally, to align with the TPU hardware configuration, certain parameters may have restrictions. Please refer to the detailed explanation below for more information.

Configuration File Parameter Description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml
   :linenos:

    base:
        seed: &seed 42
    model:
        type: Qwen2VL # Set the model name. For specific supported models, refer to the llmc/models directory.
        path: /workspace/Qwen2-VL-2B-Instruct    # Set the model weights path, please change to your desired model
        torch_dtype: auto
    calib:
        name: mme   # Set to the actual calibration dataset name, such as mme, pileval, etc.
        download: False
        path: /workspace/llmc-tpu/tpu/data/VLM/cali/MME  # Set the calibration dataset path
        n_samples: 128
        bs: 1
        seq_len: 512
        preproc: pileval_awq
        seed: *seed
    eval:
        eval_pos: [pretrain, fake_quant]
        name: mme  # Set to the actual eval dataset name, such as mme, wikitext2, etc.
        download: False
        path: /workspace/llmc-tpu/tpu/data/VLM/eval/MME # Set the eval dataset path
        bs: 1
        seq_len: 2048
    quant:
        method: Awq
        quant_objects: [language] # By default, only quantize the LLM part. If you want to quantize the VIT part, set it to [vision, language].
        weight:
            bit: 4 # Set to the desired quantization bit, supports 4 or 8
            symmetric: False # Set to False for 4-bit and True for 8-bit
            granularity: per_group # Set to per_group for 4-bit and per_channel for 8-bit.
            group_size: 64 # Set to 64 for 4-bit (corresponding to TPU-MLIR); set to -1 for 8-bit.
        special:
            trans: True
            trans_version: v2
            weight_clip: True
            clip_sym: True
    save:
        save_trans: True       # When set to True, you can save the adjusted floating-point weights.
        save_path: ./save_path # Set the path to save the weights
    run:
        task_name: awq_w_only
        task_type: VLM   # Set to VLM or LLM


The above is a complete config file constructed using the Awq algorithm as an example. To simplify user operations, users can directly copy the above into their own config and then modify the parameters that are annotated.

Below are detailed explanations of some important parameters:

.. list-table:: Introduction of Relevant Parameters
   :widths: 25 60
   :header-rows: 1

   * - Parameter
     - Description
   * - model
     - model name. the supported models are in the llmc/models directory. You can add new models by including `llmc/models/xxxx.py`.
   * - calib
     - calib class parameters mainly specify parameters related to the calibration set
   * - eval
     - eval class parameters mainly specify parameters related to the eval set.
   * - quant
     - specify the quantization parameters. It is generally recommended to use the Awq algorithm. For quant_objects, typically select language. For weight quantization parameters, refer to the table below.

To align with `TPU-MLIR`, the configuration of weight quantization related parameters is as follows:

.. list-table:: weight-only quantization parameters
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

Stage 3: Execute the Quantization Algorithm
""""""""""""""""""""""""""""""""""""""""""""""""""""

.. code-block:: shell
   :linenos:

   cd /workspace/llmc-tpu
   python3 tpu/llm_quant.py --llmc_tpu_path . --config_path ./tpu/example.yml

* config_path refers to the path of the quantization configuration file, and llmc_tpu_path refers to the current llmc_tpu directory path.
