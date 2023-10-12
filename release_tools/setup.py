from setuptools import setup
import os

setup(
    name="tpu_mlir",
    version=os.getenv("mlir_version").split("-")[0],
    author="sophgo",
    author_email="dev@sophgo.com",
    description=f"Machine learning compiler based on MLIR for Sophgo TPU {os.getenv('mlir_version')}.",
    # readme="README.md",
    license="Apache",
    platforms="unbuntu22.04",
    python_requires=">=3.10",
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
        "numpy==1.23.5",
        "scipy>=1.11.2",
        "tqdm>=4.66.1",
        "Pillow>=10.0.1",
        "plotly>=5.17.0",
        "opencv-python>=4.8.0.76",
        "protobuf==3.20.3",
        "graphviz==0.20.1",
        "pycocotools==2.0.6",
        "scikit-learn",
        "transformers==4.31.0",
    ],
    extras_require={
        "all": [
            # torch
            "torch>=2.0.1",
            "torchvision>=0.15.2",
            # onnx
            "onnx==1.14.1",
            "onnxruntime==1.15.1",
            "onnxsim==0.4.17",
            # caffe 
            "scikit-image>=0.21.0",
            "six>=1.16.0",
            # tensorflow
            "tensorflow-cpu==2.13.0",
            "tf2onnx==1.8.4",
            # paddle
            "paddlepaddle==2.5.0",
            "paddle2onnx==1.0.8",
        ],
        "torch": [
            "torch>=2.0.1",
            "torchvision>=0.15.2"
        ],
        "onnx": [
            "onnx==1.14.1",
            "onnxruntime==1.15.1",
            "onnxsim==0.4.17"
        ],
        "caffe": [
            "scikit-image>=0.21.0",
            "six>=1.16.0",
        ],
        "tensorflow": [
            "tensorflow-cpu==2.13.0",
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
            "tpu_mlir_get_resource=tpu_mlir:cp_from_package_root",
            ### Command Entries Will Be Set From Here. Do Not Delete This Line! ###
        ]
    },
)
