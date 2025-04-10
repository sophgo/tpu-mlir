from setuptools import setup
import os

setup(
    name="tpu_mlir",
    version=os.getenv("mlir_version").split("-")[0],
    author="SOPHGO",
    author_email="sales@sophgo.com",
    description=f"Machine learning compiler based on MLIR for Sophgo TPU {os.getenv('mlir_version').split('-')[0]}-g{os.getenv('mlir_commit_id')}-{os.getenv('mlir_commit_date')}",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license="2-Clause BSD",
    platforms="unbuntu22.04",
    python_requires=">=3.10,<3.11",
    url="https://github.com/sophgo/tpu-mlir",
    include_package_data=True,
    packages=["tpu_mlir"],
    keywords=["python3.10", "unbuntu22.04", "linux", "tpu-mlir"],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development",
    ],
    install_requires=[
        "numpy==1.24.3",
        "scipy==1.11.1",
        "tqdm==4.65.0",
        "Pillow==10.0.0",
        "plotly==5.15.0",
        "opencv-python-headless==4.8.0.74",
        "protobuf==3.20.3",
        "graphviz==0.20.1",
        "pycocotools==2.0.6",
        "scikit-image==0.21.0",
        "transformers==4.51.1",
        "scikit-learn==1.6.1",
        "onnxruntime_extensions==0.14.0",
        "pandas==2.0.3",
    ],
    extras_require={
        "all": [
            # torch
            "torch==2.1.0",
            "torchvision==0.16.0",
            # onnx
            "onnx==1.14.1",
            "onnxruntime==1.16.3",
            "onnxsim==0.4.17",
            # caffe
            "six==1.16.0",
            # tensorflow
            "tf2onnx==1.8.4",
            # paddle
            "paddlepaddle==2.5.0",
            "paddle2onnx==1.0.8",
            # others
            "astunparse==1.6.3",
            "flatbuffers==24.12.23",
            "setuptools==59.6.0",
            "tensorboard==2.13.0",
        ],
        "torch": [
            "torch==2.1.0",
            "torchvision==0.16.0"
        ],
        "onnx": [
            "onnx==1.14.1",
            "onnxruntime==1.16.3",
            "onnxsim==0.4.17"
        ],
        "caffe": [
            "six==1.16.0",
        ],
        "tensorflow": [
            "tf2onnx==1.8.4"
        ],
        "paddle": [
            "paddlepaddle==2.5.0",
            "paddle2onnx==1.0.8",
        ]
    },
    scripts=["release_tools/envsetup.sh"],
    entry_points={
        "console_scripts": [  # command entries corresponding to the functions in entry.py.
            # "tpu_mlir_get_resource=tpu_mlir:cp_from_package_root",
            ### Command Entries Will Be Set From Here. Do Not Delete This Line! ###
        ]
    },
)
