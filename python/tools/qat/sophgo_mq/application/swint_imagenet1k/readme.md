# Sophgo-mq Example with ImageNet

We follow the Huggingface [official example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-classification) to build the example of Model Quantization Benchmark for image classification task.

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org)) and transformers
- `pip install -r requirements.txt`

## Usage
- **Prepare the image folder:**
  Preprocess the ImageNet validation set as ImageFolder.
  ```
  cd $PATH_TO_IMAGENET_VALIDATION_SET
  sh ./application/vit_example/valprep.sh
  ```
- **Post Training Quantization:**
  We support PTQ of ViT for image-classification now.
  You can modify the config in "config.yaml". And an example quantization config is as follows. 
  ```
  quant: 
    a_qconfig:
        quantizer: FixedFakeQuantize
        observer: EMAMSEObserver
        bit: 8
        symmetric: False
        ch_axis: -1
    w_qconfig:
        quantizer: FixedFakeQuantize
        observer: MinMaxObserver
        bit: 8
        symmetric: True
        ch_axis: 0
    calibrate: 1024
    backend: academic
  ```
  You need to prepare the finetuned FP32 model of the specific task and change the "model_name_or_path" in "config.yaml".
  
  Steps:
    ```
    git clone https://github.com/sophgo/sophgo-mq.git
    cd application/vit_example
    sh run.sh
    ```

## Results

| Model                                                                                               | Results@FP32            | Results@int8            |
| :-------------------------------------------------------------------------------------------------- | :-----------------------| :-----------------------|
| [**vit-base-patch16-224**](https://huggingface.co/google/vit-base-patch16-224)                      | ACC@1 81.31 ACC@5 95.94 | ACC@1 81.03 ACC@5 95.78 |
| [**vit-large-patch16-224**](https://huggingface.co/google/vit-large-patch16-224)                    | ACC@1 82.52 ACC@5 96.35 | ACC@1 82.59 ACC@5 96.34 |
| [**swin-tiny-patch4-window7-224**](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)   | ACC@1 80.93 ACC@5 95.46 | ACC@1 80.77 ACC@5 95.40 |
| [**swin-base-patch4-window7-224**](https://huggingface.co/microsoft/swin-base-patch4-window7-224)   | ACC@1 84.81 ACC@5 97.43 | ACC@1 84.75 ACC@5 97.40 |
| [**swin-large-patch4-window7-224**](https://huggingface.co/microsoft/swin-large-patch4-window7-224) | ACC@1 86.02 ACC@5 97.85 | ACC@1 86.00 ACC@5 97.86 |
