# FP8 Emulation Toolkit
## Introduction
This repository provides PyTorch tools to emulate the new `FP8` formats on top of existing floating point hardware from Intel (FP32) and NVIDIA (FP16). In addition to the two formats `E5M2` and `E4M3` defined in the joint specification from <a href=https://arxiv.org/pdf/2209.05433https://arxiv.org/pdf/2209.05433 target="_blank">ARM-Intel-NVIDIA</a>, the toolkit also suports a third variant named `E3M4` which follows the guidelines established for `E4M3` format.

Following table shows the binary formats and the numeric range:

|                |                                 E5M2                             |                               E4M3                               |                                E3M4                               |
| -------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------- |  ---------------------------------------------------------------- |
| Exponent Bias  |                                 15                               |                                  7                               |                                   3                               |
| Infinities     | S.11111.00<sub>2</sub>                                           |                                 N/A                              |                                  N/A                              |
| NaNs           | S.11111.{01, 10, 11}<sub>2</sub>                                 | S.1111.111<sub>2</sub>                                           |  S.111.1111<sub>2</sub>                                           |
| Zeros          | S.00000.00<sub>2</sub>                                           | S.0000.000<sub>2</sub>                                           |  S.000.0000<sub>2</sub>                                           |
| Max normal     | S.11110.11<sub>2</sub>=1.75 * 2<sup>15</sup>=57344.0             | S.1111.110<sub>2</sub>=1.75 * 2<sup>8</sup>=448.0                |  S.111.1110<sub>2</sub>=1.875 * 2<sup>4</sup>=30.0                |
| Min normal     | S.00001.00<sub>2</sub>=2<sup>-14</sup>=6.1e<sup>-05</sup>        | S.0001.000<sub>2</sub>=2<sup>-6</sup>=1.5e<sup>-02</sup>         |  S.001.0000<sub>2</sub>=2<sup>-2</sup>=2.5e<sup>-01</sup>         |
| Max subnormal  | S.00000.11<sub>2</sub>=0.75 * 2<sup>-14</sup>=4.5e<sup>-05</sup> | S.0000.111<sub>2</sub>=0.875 * 2<sup>-6</sup>=1.3e<sup>-02</sup> |  S.000.1111<sub>2</sub>=0.9375 * 2<sup>-2</sup>=2.3e<sup>-01</sup> |
| Min subnormal  | S.00000.01<sub>2</sub>=2<sup>-16</sup>=1.5e<sup>-05</sup>        | S.0000.001<sub>2</sub>=2<sup>-9</sup>=1.9e<sup>-03</sup>         |  S.000.0001<sub>2</sub>=2<sup>-6</sup>=1.5e<sup>-02</sup>         |

![DataFormats](./docs/formats.png)

## Installation

Follow the instructions below to install FP8 Emulation Toolkit in a Python virtual environment.
Alternatively, this installation can also be performed in a docker environment.

### Requirements
Install or upgrade the following packages on your linux machine.

* Python >= 3.8.5
* CUDA >= 11.1
* gcc >= 8.4.0

Make sure these versions are reflected in the `$PATH`

#### Target Hardware
* CPU >= Icelake Xeon
* GPU >= V100

### Create a Python virtual environment
```
$ python3 -m ~/py-venv
$ cd ~/py-venv
$ source bin/activate
$ pip3 install --upgrade pip3
```
### Clone and install FP8 Emulation Toolkit
```
$ git clone https://github.com/IntelLabs/FP8-Emulation-Toolkit.git
$ cd FP8-Emulation-Toolkit
$ pip3 install -r requirements.txt
$ python setup.py install
```

## Usage Examples
The emulated FP8 formats can be experimented with by integrated them into standard deep learning flows. Follow the links below for detailed instructions and code samples for exploring training and inference flows using FP8 data formats. 

* [Post-training quantization](./examples/inference) 
* [Mixed precision training](./examples/training) 


## Related Work
This implementation is based on the following research. Check out the source material for more details on the training and inference methods. 

```
@misc{mellempudi2019mixed,
      title={Mixed Precision Training With 8-bit Floating Point}, 
      author={Naveen Mellempudi and Sudarshan Srinivasan and Dipankar Das and Bharat Kaul},
      year={2019},
      eprint={1905.12334},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
```
@misc{micikevicius2022fp8,
      title={FP8 Formats for Deep Learning}, 
      author={Paulius Micikevicius and Dusan Stosic and Neil Burgess and Marius Cornea and Pradeep Dubey and Richard Grisenthwaite and Sangwon Ha and Alexander Heinecke and Patrick Judd and John Kamalu and Naveen Mellempudi and Stuart Oberman and Mohammad Shoeybi and Michael Siu and Hao Wu},
      year={2022},
      eprint={2209.05433},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
