# Sophgo-mq Example with Glue

We follow the Huggingface [official example][https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification] to build the example of Model Quantization Benchmark for text classification task.

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org)) and transformers
- `pip install -r requirements.txt`

## Usage

- **Post Training Quantization:**
  We support PTQ of Bert for text-classification now.
  完成了Bert的text-classification的PTQ，如果选择INT类型的fake quantizer可以在config.yaml里进行更改；如果选择FP8的fake quantizer可以在config-fp8.yaml里更改
  You can modify the config in "config.yaml". And an example quantization config is as follows. 
  ```
  quant:
    a_qconfig:
      quantizer: FixedFakeQuantize
      observer: EMAQuantileObserver
      bit: 8
      symmetric: False
      per_channel: False
    w_qconfig:
      quantizer: FixedFakeQuantize
      observer: MinMaxObserver
      bit: 8
      symmetric: True
      per_channel: True
    calibrate: 64
    pot_scale: False
    backend: academic
  ```
  You need to prepare the finetuned FP32 model of the specific task and change the "model_name_or_path" in "config.yaml" or "config-fp8.yaml".
  
  Steps:
    ```
    git clone https://github.com/sophgo/sophgo-mq.git
    cd application/nlp_example
    sh run.sh 运行INT类型NLP PTQ
    sh run-fp8.sh 运行FP8 NLP PTQ
    ```

## Results

| Task             | Metrics                    | results@FP32                         | results@int8                  |
| :--------------- | :------------------------- | :----------------------------------- | :---------------------------- |
| **mrpc**         | **acc/f1**                 | **87.75/91.35**                      | **87.75/91.2**                |
| **mnli**         | **acc m/mm**               | **84.94/84.76**                      | **84.69/84.59**               |
| **cola**         | **Matthews corr**          | **59.6**                             | **59.41**                     |
| **sst2**         | **acc**                    | **93.35**                            | **92.78**                     |
| **stsb**         | **Pearson/Spearman corr**  | **89.70/89.28**                      | **89.36/89.22**               |
| **qqp**          | **f1/acc**                 | **87.82/90.91**                      | **87.46/90.72**               |
| **rte**          | **acc**                    | **72.56**                            | **71.84**                     |
| **qnli**         | **acc**                    | **91.84**                            | **91.32**                     |

