FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

MAINTAINER Maintainer

EXPOSE 3000

ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

RUN apt update -yq && \
    apt install -yq curl wget unzip git vim cmake zlib1g-dev g++ gcc sudo

RUN apt install -yq build-essential libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev

RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.7
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.7 get-pip.py

RUN adduser --disabled-password --gecos '' docker && \
    adduser docker sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN mkdir -p /.cache/pip
RUN mkdir -p /.local/share
RUN chown -R docker:docker /.cache/pip
RUN chown -R docker:docker /.local

USER docker 

WORKDIR /home/docker/

ENV PATH="/home/docker/.local/bin:${PATH}"
RUN chown -R docker:docker /home/docker/

RUN python3.7 -m pip install networkx==2.3 seaborn==0.9.0 ipdb==0.12 tensorboard_logger==0.1.0 
RUN python3.7 -m pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html 
RUN python3.7 -m pip install numpy==1.19.4 tqdm==4.31.1 matplotlib==3.2.1 pulp==2.4 PyYAML==5.4.1 scikit_learn==0.24.1 torch_geometric==1.6.3 sacred==0.8.2
RUN python3.7 -m pip install https://pytorch-geometric.com/whl/torch-1.7.0+cu101/torch_sparse-0.6.8-cp37-cp37m-linux_x86_64.whl
RUN python3.7 -m pip install https://pytorch-geometric.com/whl/torch-1.7.0+cu101/torch_scatter-2.0.5-cp37-cp37m-linux_x86_64.whl
RUN python3.7 -m pip install tensorboard==2.3.0
RUN python3.7 -m pip uninstall dataclasses -y
RUN python3.7 -m pip install jupyter
RUN pwd

ENV PYTHONPATH "${PYTHONPATH}:/home/docker/.local/lib/python3.7/site-packages/"
RUN chown -R docker:docker /home/docker/.local/
RUN chmod -R 777 /home/docker/.local/
RUN chmod -R 777 /.cache/pip/
RUN python3.7 -m pip list

USER docker
WORKDIR /home/docker/app