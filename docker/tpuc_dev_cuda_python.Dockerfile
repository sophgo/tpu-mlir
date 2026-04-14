FROM sophgo/tpuc_dev:v3.4
ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ=Asia/Shanghai

RUN apt update && apt install -y bc zlib1g gnupg software-properties-common curl

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb && apt update && rm -rf cuda-keyring_1.1-1_all.deb

RUN apt install -y cuda-toolkit-12-6
ENV PATH=$PATH:/usr/local/cuda-12.6/bin
ENV CPATH=$CPATH:/usr/local/cuda-12.6/include
ENV LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-12.6/lib64
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.6/lib64

RUN wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.12.0.46_cuda12-archive.tar.xz && \
    tar -xJf ./cudnn-linux-x86_64-9.12.0.46_cuda12-archive.tar.xz && \
    cp -av cudnn-linux-x86_64-9.12.0.46_cuda12-archive/include/* /usr/local/cuda-12.6/include && \
    cp -av cudnn-linux-x86_64-9.12.0.46_cuda12-archive/lib/*  /usr/local/cuda-12.6/lib64 && \
    rm -rf cudnn-linux-x86_64-9.12.0.46_cuda12-archive && ldconfig

RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.5.0/tars/TensorRT-10.5.0.18.Linux.x86_64-gnu.cuda-12.6.tar.gz && \
    tar -xzf TensorRT-10.5.0.18.Linux.x86_64-gnu.cuda-12.6.tar.gz && \
    mv TensorRT-10.5.0.18 /opt/tensorrt-10.5 && \
    rm -rf TensorRT-10.5.0.18.Linux.x86_64-gnu.cuda-12.6.tar.gz
ENV LD_LIBRARY_PATH=/opt/tensorrt-10.5/lib:$LD_LIBRARY_PATH
RUN ldconfig

WORKDIR /workspace
COPY requirements_cuda.txt .
RUN pip3 install -r requirements_cuda.txt --extra-index-url https://download.pytorch.org/whl/cu126 && \
    rm -rf ~/.cache/pip/* requirements_cuda.txt

ENV LC_ALL=C.UTF-8
