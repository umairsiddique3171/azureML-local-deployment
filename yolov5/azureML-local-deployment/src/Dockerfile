FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*
