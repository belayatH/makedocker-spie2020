FROM nvidia/cuda:10.0-cudnn7-runtime

LABEL maintainer "example@example.jp"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get -y install python3-pip curl \
    python3-dev \
    python3-pip \
    libopenmpi-dev \
    libopencv-dev \
    vim \
    wget \
    && \
    apt-get clean && \
    rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*

RUN pip3 install keras tensorflow-gpu scikit-learn pillow \
    matplotlib tqdm opencv-python hmmlearn pandas h5py

#Copy current-dir data to /tmp/work
COPY . /root/
