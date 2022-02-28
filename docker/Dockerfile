FROM ubuntu:20.04
LABEL maintainer pengchao.hu@sophgo.com

# ================== select apt source ==================
COPY sources.list.aliyun /etc/apt/sources.list.aliyun

RUN cp /etc/apt/sources.list /etc/apt/sources.list.origin; \
    cp /etc/apt/sources.list.aliyun /etc/apt/sources.list; \

# ================== zoneinfo ==================
RUN apt-get update \
    &&  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

RUN TZ=Asia/Shanghai \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata

# ================== apt install ==================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gdb ninja-build automake autoconf libtool \
    vim wget curl git parallel ssh-client unzip zip tree python3-dev \
    python3-pip jq default-jre rsync device-tree-compiler libssl-dev \
    ssh bc slib cpio squashfs-tools fakeroot pkg-config autopoint \
    bison flex gettext indent meson libglib2.0-dev clang-format \
    libboost-all-dev libgflags-dev libgoogle-glog-dev libhdf5-serial-dev \
    libleveldb-dev liblmdb-dev libprotobuf-dev libsnappy-dev \
    protobuf-compiler libopenblas-dev

# for 32bit arch
RUN dpkg --add-architecture i386
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    libc6:i386 libncurses5:i386 libstdc++6:i386

# for cmake
RUN apt-get install -y --no-install-recommends \
    apt-transport-https ca-certificates gnupg \
    software-properties-common
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake-data cmake

# cleanup
RUN rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip; \
    pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple; \
    pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple; \

RUN pip3 install --no-cache-dir --upgrade setuptools
RUN pip3 install --no-cache-dir \
    Cython numpy scipy==1.4.1 scikit-image matplotlib \
    wheel leveldb decorator networkx nose python-dateutil \
    protobuf pyyaml Pillow six lmdb pathlib tqdm enum34 \
    termcolor opencv-python-headless argcomplete jupyterlab \
    paramiko jedi==0.17.2 ipython torch==1.10.2 torchvision==0.11.2 \
    onnx==1.10.2 onnxruntime==1.10.0 onnx-simplifier \
    tensorflow==2.6.2 packaging==20.3 Jinja2==2.11.2 \
    tf2onnx==1.9.3 pycocotools ply gitpython "pybind11[global]"

#=================== Setup default python =========
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 0
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 0

# ================== Install repo v2.8 ==================
ADD files/repo /usr/bin
ADD files/agent.jar /root

# ================= Cleanup =================
RUN rm -rf /tmp/*

WORKDIR /work/
