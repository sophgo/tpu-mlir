FROM ubuntu:22.04 as base
ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ=Asia/Shanghai

# ********************************************************************************
#
# stage 0: base
# ********************************************************************************

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https ca-certificates build-essential \
    git vim sudo \
    # python
    python3-dev \
    python3-venv \
    python3-pip \
    virtualenv \
    swig \
    tzdata \
    # tablgen
    libncurses5-dev libncurses5 \
    # tools
    ninja-build \
    parallel \
    curl wget \
    unzip \
    graphviz \
    bsdmainutils \
    gdb \
    ccache \
    git-lfs \
    clang lld lldb clang-format \
    libomp-dev \
    # for firmware
    flex bison \
    # for opencv
    libgl1 \
    libnuma1 libatlas-base-dev \
    # ssh
    openssh-server openssh-client \
    # release package
    patchelf \
    # for document
    texlive-xetex \
    # fix bug: https://bugs.archlinux.org/task/67856
    texlive-lang-chinese texlive-fonts-recommended \
    # for regression
    sshpass \
    rsync && \
    # for paddlePaddle
    wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb && \
    dpkg -i libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb && \
    rm libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb && \
    # cleanup
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

ENV CMAKE_VERSION 3.25.3

RUN wget https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-linux-x86_64.sh \
    --no-check-certificate \
    -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && /tmp/cmake-install.sh --skip-license --prefix=/usr/local \
    && rm /tmp/cmake-install.sh

ENV CATCH3_VERSION 3.4.0

RUN git clone --branch v${CATCH3_VERSION} https://github.com/catchorg/Catch2.git && \
    cd Catch2 && \
    cmake -Bbuild -H. -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build build/ --target install && \
    rm -rf /Catch2 /tmp/* ~/.cache/*

ARG FLATBUFFERS_VERSION="e2be0c0b0605b38e11a8710a8bae38d7f86a7679"
RUN git clone https://github.com/google/flatbuffers.git &&\
    cd flatbuffers && git checkout ${FLATBUFFERS_VERSION} && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build . --target install && \
    cd / && rm -rf flatbuffers /tmp/* ~/.cache/*

ARG ONEDNN_VERSION="5aabea153825347afa92a2d9f69dd893246bea45"
RUN git clone https://github.com/oneapi-src/oneDNN.git && \
    cd oneDNN && git checkout ${ONEDNN_VERSION} && \
    mkdir -p build && cd build && \
    cmake -G Ninja .. \
    -DDNNL_CPU_RUNTIME=OMP \
    -DDNNL_BUILD_TESTS=OFF \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build . --target install && \
    cd / && rm -rf oneDNN /tmp/* ~/.cache/*

# MLIR&Caffe python dependency
RUN pip install pybind11-global==2.11.1 numpy==1.24.3 PyYAML==5.4.1 && \
    rm -rf ~/.cache/pip/*

# ********************************************************************************
#
# stage 1: caffe
# ********************************************************************************

FROM base as caffe_builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libprotobuf-dev \
    libsnappy-dev \
    protobuf-compiler \
    libopenblas-dev
# RUN pip install numpy
WORKDIR /root

ARG CAFFE_VERSION="6b665ea602f602502121ca3dc84f58e801fcaa13"
RUN git clone https://github.com/sophgo/caffe.git && \
    cd caffe && git checkout ${CAFFE_VERSION} && \
    mkdir -p build && cd build && \
    cmake -G Ninja .. \
    -DCPU_ONLY=ON -DUSE_OPENCV=OFF \
    -DBLAS=open -DUSE_OPENMP=TRUE \
    -DCMAKE_CXX_FLAGS=-std=gnu++11 \
    -Dpython_version="3" \
    -DCMAKE_INSTALL_PREFIX=caffe && \
    cmake --build . --target install
RUN cd /root/caffe/python/caffe && \
    rm _caffe.so && \
    cp -f /root/caffe/build/lib/_caffe.so . && \
    cp -rf /root/caffe/src/caffe/proto .

# ********************************************************************************
#
# stage 2: MLIR
# ********************************************************************************
FROM base as mlir_builder

ARG LLVM_VERSION="c67e443895d5b922d1ffc282d23ca31f7161d4fb"
RUN git clone https://github.com/llvm/llvm-project.git && \
    cd llvm-project/ && \
    git checkout ${LLVM_VERSION} && \
    mkdir build && cd build && \
    cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_TARGETS_TO_BUILD="" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_INCLUDE_TESTS=OFF \
    -DLLVM_INSTALL_GTEST=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON && \
    cmake --build . --target install && \
    cd / && rm -rf llvm-project /tmp/* ~/.cache/*

RUN TZ=Asia/Shanghai \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata \
    # install some fonts
    && wget "http://mirrors.ctan.org/fonts/fandol.zip" -O /usr/share/fonts/fandol.zip \
    && unzip /usr/share/fonts/fandol.zip -d /usr/share/fonts \
    && rm /usr/share/fonts/fandol.zip \
    && git config --global --add safe.directory '*' \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring_1.1-1_all.deb \
    && rm -rf /tmp/*



# ********************************************************************************
#
# stage 2: final
# ********************************************************************************
FROM mlir_builder as final

RUN apt-get update && apt-get install -y --no-install-recommends \
    # caffe dependency
    libboost-python1.74.0 \
    libboost-filesystem1.74.0 \
    libboost-system1.74.0 \
    libboost-regex1.74.0 \
    libboost-thread1.74.0 \
    libgoogle-glog0v5 \
    libopenblas0 \
    libprotobuf23 \
    libgflags-dev && \
    # clenup
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=caffe_builder /root/caffe/python/caffe /usr/local/python_packages/caffe

ENV LC_ALL=C.UTF-8
WORKDIR /workspace
