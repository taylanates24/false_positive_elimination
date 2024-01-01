FROM nvcr.io/nvidia/pytorch:23.05-py3

ENV DEBIAN_FRONTEND=noninteractive

ENV MPLBACKEND=agg

RUN apt-get update && \
        apt-get install -y \
        git \
        wget \
        unzip \
        vim \
        zip \
        curl \
        yasm \
        pkg-config \
        nano \
        tzdata \
        ffmpeg \
        libgtk2.0-dev \
        libgl1-mesa-glx && \
    rm -rf /var/cache/apk/*

RUN pip install --upgrade pip

RUN pip --no-cache-dir install \
      Cython==0.29.21

RUN pip --no-cache-dir install \
      numpy \
	matplotlib \
	tqdm \
	imageio \
        pillow \
	opencv-python==4.5.5.64 \
	tensorboard \
	pyyaml \
      neptune-client \
      pytorch-lightning==1.9.2

RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
       cd torch2trt && python3 setup.py install

RUN pip3 install timm==0.6.12
RUN pip install imgaug==0.4.0
RUN ln -sf /usr/share/zoneinfo/Turkey /etc/localtime

