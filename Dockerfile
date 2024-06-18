FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3 python3-pip ffmpeg libsm6 libxext6 && \
    pip3 install --upgrade pip setuptools

WORKDIR /logo-detection

COPY . /logo-detection

RUN pip3 install -r requirements.txt
