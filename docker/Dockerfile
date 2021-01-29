FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

MAINTAINER Wei Qiu

RUN apt update -yq && \
    apt install -yq curl wget unzip git vim cmake zlib1g-dev g++ gcc sudo

RUN apt install -y software-properties-common
RUN apt install -y python3.7
RUN apt install -y python3-pip
RUN python3.7 -m pip install --upgrade pip
RUN which python3

RUN adduser --disabled-password --gecos '' docker && \
    adduser docker sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER docker 

WORKDIR /home/docker/


RUN python3.7 -m pip install networkx==2.3 seaborn==0.9.0 ipdb==0.12 tensorboard_logger==0.1.0 torch==1.7.0  torchvision==0.8.1
RUN python3.7 -m pip install numpy==1.19.4 tqdm==4.31.1 matplotlib==3.2.1 pulp==2.4 PyYAML==5.4.1 scikit_learn==0.24.1 torch_geometric==1.6.3 sacred==0.8.2

RUN chown -R docker:docker /home/docker/
ENV PATH="/home/docker/.local/bin:${PATH}"

USER docker
WORKDIR /home/docker/app
