附录01：各框架模型转ONNX参考
==================================

本章节主要提供了将PyTorch, TensorFlow与PaddlePaddle模型转为ONNX模型的方式参考，读者也可以参考ONNX官方仓库提供的转模型教程： https://github.com/onnx/tutorials。本章节中的所有操作均在Docker容器中进行，具体的环境配置方式请参考第二章的内容。

PyTorch模型转ONNX
-----------------------
本节以一个自主搭建的简易PyTorch模型为例进行onnx转换

步骤0：创建工作目录
~~~~~~~~~~~~~~~~~~~~~~~

在命令行中创建并进入torch_model目录。

.. code-block:: shell
   :linenos:

   $ mkdir torch_model
   $ cd torch_model

步骤1：搭建并保存模型
~~~~~~~~~~~~~~~~~~~~~~~

在该目录下创建名为 ``simple_net.py`` 的脚本并运行，脚本的具体内容如下：

.. code-block:: python3
   :linenos:

   #!/usr/bin/env python3
   import torch

   # Build a simple nn model
   class SimpleModel(torch.nn.Module):

      def __init__(self):
         super(SimpleModel, self).__init__()
         self.m1 = torch.nn.Conv2d(3, 8, 3, 1, 0)
         self.m2 = torch.nn.Conv2d(8, 8, 3, 1, 1)

      def forward(self, x):
         y0 = self.m1(x)
         y1 = self.m2(y0)
         y2 = y0 + y1
         return y2

   # Create a SimpleModel and save its weight in the current directory
   model = SimpleModel()
   torch.save(model.state_dict(), "weight.pth")

运行命令如下：

.. code-block:: shell

   $ python simple_net.py

运行完后我们会在当前目录下获得一个 ``weight.pth`` 的权重文件。

步骤2：导出ONNX模型
~~~~~~~~~~~~~~~~~~~~~~

在该目录下创建另一个名为 ``export_onnx.py`` 的脚本并运行，脚本的具体内容如下：

.. code-block:: python3
   :linenos:

   #!/usr/bin/env python3
   import torch
   from simple_net import SimpleModel

   # Load the pretrained model and export it as onnx
   model = SimpleModel()
   model.eval()
   checkpoint = torch.load("weight.pth", map_location="cpu")
   model.load_state_dict(checkpoint)

   # Prepare input tensor
   input = torch.randn(1, 3, 16, 16, requires_grad=True)

   # Export the torch model as onnx
   torch.onnx.export(model,
                     input,
                     'model.onnx', # name of the exported onnx model
                     opset_version=13,
                     export_params=True,
                     do_constant_folding=True)

运行完脚本后，我们即可在当前目录下得到名为 ``model.onnx`` 的onnx模型。

TensorFlow模型转ONNX
-----------------------

本节以TensorFlow官方仓库中提供的 ``mobilenet_v1_0.25_224`` 模型作为转换样例。

步骤0：创建工作目录
~~~~~~~~~~~~~~~~~~~~~~~

在命令行中创建并进入tf_model目录。

.. code-block:: shell
   :linenos:

   $ mkdir tf_model
   $ cd tf_model

步骤1：准备并转换模型
~~~~~~~~~~~~~~~~~~~~~~

命令行中通过以下命令下载模型并利用 ``tf2onnx`` 工具将其导出为ONNX模型：

.. code-block:: shell
   :linenos:

   $ wget -nc http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224.tgz
   # tar to get "*.pb" model def file
   $ tar xzf mobilenet_v1_0.25_224.tgz
   $ python -m tf2onnx.convert --graphdef mobilenet_v1_0.25_224_frozen.pb \
       --output mnet_25.onnx --inputs input:0 \
       --inputs-as-nchw input:0 \
       --outputs MobilenetV1/Predictions/Reshape_1:0

运行以上所有命令后我们即可在当前目录下得到名为 ``mnet_25.onnx`` 的onnx模型。


PaddlePaddle模型转ONNX
------------------------

本节以PaddlePaddle官方仓库中提供的SqueezeNet1_1模型作为转换样例。
本节需要额外安装openssl-1.1.1o（ubuntu 22.04默认提供openssl-3.0.2）。

步骤0：安装openssl-1.1.1o
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell
   :linenos:

   wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
   sudo dpkg -i libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb


如果上述链接失效，请参考 http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/?C=M;O=D 更换有效链接.

步骤1：创建工作目录
~~~~~~~~~~~~~~~~~~~~~~~

在命令行中创建并进入pp_model目录。

.. code-block:: shell
   :linenos:

   $ mkdir pp_model
   $ cd pp_model

步骤2：准备模型
~~~~~~~~~~~~~~~~~~~~~~

在命令行中通过以下命令下载模型：

.. code-block:: shell
   :linenos:

   $ wget https://bj.bcebos.com/paddlehub/fastdeploy/SqueezeNet1_1_infer.tgz
   $ tar xzf SqueezeNet1_1_infer.tgz
   $ cd SqueezeNet1_1_infer

并用PaddlePaddle项目中的 ``paddle_infer_shape.py`` 脚本对模型进行shape推理，此处将输入shape以NCHW的格式设置为 ``[1,3,224,224]`` ：

.. code-block:: shell
   :linenos:

   $ wget https://raw.githubusercontent.com/jiangjiajun/PaddleUtils/main/paddle/paddle_infer_shape.py
   $ python paddle_infer_shape.py  --model_dir . \
                             --model_filename inference.pdmodel \
                             --params_filename inference.pdiparams \
                             --save_dir new_model \
                             --input_shape_dict="{'inputs':[1,3,224,224]}"

运行完以上所有命令后我们将处于 ``SqueezeNet1_1_infer`` 目录下，并在该目录下生成 ``new_model`` 的目录。

步骤3：转换模型
~~~~~~~~~~~~~~~~~~~~~

在命令行中通过以下命令安装 ``paddle2onnx`` 工具，并利用该工具将PaddlePaddle模型转为ONNX模型：

.. code-block:: shell
   :linenos:

   $ pip install paddle2onnx
   $ paddle2onnx  --model_dir new_model \
             --model_filename inference.pdmodel \
             --params_filename inference.pdiparams \
             --opset_version 13 \
             --save_file squeezenet1_1.onnx

运行完以上所有命令后我们将获得一个名为 ``squeezenet1_1.onnx`` 的onnx模型。
