import os
import torch
import setuptools
from sophgo_mq import __version__
from torch.utils.cpp_extension import BuildExtension, CppExtension


def read_requirements():
    reqs = []
    with open('requirements.txt', 'r') as fin:
        for line in fin.readlines():
            reqs.append(line.strip())
    return reqs


cmdclass = {}
ext_modules = []

ext_modules.append(
    CppExtension(
        'fpemu_cpp',
        ['FP8_Emulator/pytquant/cpp/avx-fpemu.cpp'
         ],  # If the machine supports AVX-512, add the 512 instruction set compilation arguments below and replace avx-fpemu with fpemu_impl.cpp.
        extra_compile_args=["-mf16c", "-mavx2", "-mlzcnt", "-fopenmp",
                            "-Wdeprecated-declarations"]), )

if torch.cuda.is_available():
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    ext_modules.append(
        CUDAExtension(
            'fpemu_cuda',
            [
                'FP8_Emulator/pytquant/cuda/fpemu_impl.cpp',
                'FP8_Emulator/pytquant/cuda/fpemu_kernels.cu'
            ],
        ), )
cmdclass['build_ext'] = BuildExtension

setuptools.setup(name="sophgo_mq",
                 version=__version__,
                 author="SOPHGO Technology Co. Ltd.",
                 author_email="",
                 description=("PTQ and Quantization aware training for SOPHGO devices."),
                 ext_modules=ext_modules,
                 cmdclass=cmdclass,
                 url="",
                 python_requires='>=3.6',
                 packages=setuptools.find_packages(),
                 classifiers=('Development Status :: 3 - Alpha',
                              "Programming Language :: Python :: 3",
                              "Operating System :: POSIX :: Linux :: OS Independent"),
                 install_requires=read_requirements())
