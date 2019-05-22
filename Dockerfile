FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

# Prefetch ai2thor data
RUN apt-get update && apt-get -y install wget unzip 

WORKDIR /app/data

RUN wget http://vision.stanford.edu/yukezhu/thor_v1_scene_dumps.zip
RUN unzip thor_v1_scene_dumps.zip 

WORKDIR /app

COPY requirements.txt /app
# Prefetch: install packages to previous layers
RUN python -m pip install -r /app/requirements.txt

COPY . /app

RUN python -c 'from agent.resnet import resnet50; resnet50(pretrained=True)'