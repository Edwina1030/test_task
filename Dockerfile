FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime 
RUN pip install SimpleITK
COPY . /opt/app
WORKDIR /opt/app

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev libglib2.0-0

RUN pip install -r requirements.txt
RUN pip install pandas