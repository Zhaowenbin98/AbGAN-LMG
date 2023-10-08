# hash:sha256:3954c13ce69a72aa9a2d5276dec55beeb7f576ac68185967683a8c89be440a6c
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    numpy==1.18.1 \
    pandas==1.0.1 \
    tensorflow-gpu==2.2.0
