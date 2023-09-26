from setuptools import setup
import os

setup(
    name="tpu_mlir",
    version=os.getenv("mlir_version").split("-")[0],
    author="sophgo",
    author_email="dev@sophgo.com",
    description="tpu-mlir release packages for pip install",
    readme="README.md",
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
        "numpy>=1.26.0",
        "scipy>=1.11.2",
        "tqdm>=4.66.1",
        "Pillow>=10.0.1",
        "plotly>=5.17.0",
        "opencv-python>=4.8.0.76",
    ],
    extras_require={
        "all": [
            # torch
            "torch>=2.0.1",
            "torchvision>=0.15.2",
            # onnx
            "onnx==1.14.1",
            "onnxruntime==1.15.1",
            "onnxsim==0.4.17"
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
    },
    scripts=["release_tools/envsetup.sh"],
    entry_points={
        "console_scripts": [  # command entries corresponding to the functions in entry.py.
            ### Command Entries Will Be Set From Here. Do Not Delete This Line! ###
        ]
    },
)
