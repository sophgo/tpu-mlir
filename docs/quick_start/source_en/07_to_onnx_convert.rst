Reference for converting model to ONNX format
=============================================

This chapter provides a reference for how to convert PyTorch, TensorFlow and PaddlePaddle models to ONNX format. You can also refer to the model conversion tutorial provided by ONNX official repository: https://github.com/onnx/tutorials. All the operations in this chapter are carried out in the Docker container. For the specific environment configuration method, please refer to the content of Chapter 2.

PyTorch model to ONNX
----------------------
This section takes a self-built simple PyTorch model as an example to perform onnx conversion.

Step 0: Create a working directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create and enter the torch_model directory using the command line.

.. code-block:: shell
   :linenos:

   $ mkdir torch_model
   $ cd torch_model

Step 1: Build and save the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a script named simple_net.py in this directory and run it. The specific content of the script is as follows:

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

After running the script, we will get a weight.pth weight file in the current directory.

Step 2: Export ONNX model
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create another script named export_onnx.py in the same directory and run it. The specific content of the script is as follows:

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

After running the script, we can get the onnx model named model.onnx in the current directory.

TensorFlow model to ONNX
-------------------------

In this section, we use the mobilenet_v1_0.25_224 model provided in the TensorFlow official repository as a conversion example.

Step 0: Create a working directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create and enter the tf_model directory using the command line.

.. code-block:: shell
   :linenos:

   $ mkdir tf_model
   $ cd tf_model

Step 1: Prepare and convert the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the model with the following commands and use the tf2onnx tool to export it as an ONNX model:

.. code-block:: shell
   :linenos:

   $ wget -nc http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224.tgz
   # tar to get "*.pb" model def file
   $ tar xzf mobilenet_v1_0.25_224.tgz
   $ python -m tf2onnx.convert --graphdef mobilenet_v1_0.25_224_frozen.pb \
       --output mnet_25.onnx --inputs input:0 \
       --outputs MobilenetV1/Predictions/Reshape_1:0

After running all commands, we can get the onnx model named mnet_25.onnx in the current directory.

Step 2: Modify the Input Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The input format of the TensorFlow model is NHWC by default even after it is converted to the ONNX format. An additional transpose operator is connected to the input to convert the format to NCHW. Therefore, we need to run the following python script to modify the input format to NCHW and remove that transpose operator.

.. code-block:: python3
   :linenos:

   #!/usr/bin/env python3
   import onnx

   model = onnx.load('mnet_25.onnx')
   print(model.graph.input[0].type.tensor_type.shape.dim)
   model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 3 # channel
   model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 224 # height
   model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 224 # width
   print(model.graph.input[0].type.tensor_type.shape.dim)
   input_name = model.graph.input[0].name
   del model.graph.node[0]
   model.graph.node[0].input[0] = input_name
   onnx.save(model, 'mnet_25_new.onnx')

After running the script, we can get the onnx model named mnet_25_new.onnx in the current directory.

PaddlePaddle model to ONNX
---------------------------

This section uses the SqueezeNet1_1 model provided in the official PaddlePaddle repository as a conversion example.

Step 0: Create a working directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create and enter the pp_model directory using the command line.

.. code-block:: shell
   :linenos:

   $ mkdir pp_model
   $ cd pp_model

Step 1: Prepare the model
~~~~~~~~~~~~~~~~~~~~~~~~~

Download the model with the following commands:

.. code-block:: shell
   :linenos:

   $ wget https://bj.bcebos.com/paddlehub/fastdeploy/SqueezeNet1_1_infer.tgz
   $ tar xzf SqueezeNet1_1_infer.tgz
   $ cd SqueezeNet1_1_infer

In addition, use the paddle_infer_shape.py script from the PaddlePaddle project to perform shape inference on the model. The input shape is set to [1,3,224,224] in NCHW format here:

.. code-block:: shell
   :linenos:

   $ wget https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/tools/paddle/paddle_infer_shape.py
   $ paddle_infer_shape.py  --model_dir . \
                             --model_filename inference.pdmodel \
                             --params_filename inference.pdiparams \
                             --save_dir new_model \
                             --input_shape_dict="{'inputs':[1,3,224,224]}"

After running all commands, we will be in the SqueezeNet1_1_infer directory, and there will be a new_model directory under this directory.

Step 2: Convert the model
~~~~~~~~~~~~~~~~~~~~~~~~~

Install the paddle2onnx tool through the following commands, and use this tool to convert the PaddlePaddle model to the ONNX format:

.. code-block:: shell
   :linenos:

   $ pip install paddle2onnx
   $ paddle2onnx  --model_dir new_model \
             --model_filename inference.pdmodel \
             --params_filename inference.pdiparams \
             --opset_version 13 \
             --save_file squeezenet1_1.onnx

After running all the above commands we will get an onnx model named squeezenet1_1.onnx.

