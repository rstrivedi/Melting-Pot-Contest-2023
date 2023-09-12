ARG PYTHON_VERSION=3.10
ARG CUDA_VERSION=11.6.2
ARG UBUNTU_VERSION=20.04

FROM nvidia/cuda:${CUDA_VERSION}}-devel-ubuntu${UBUNTU_VERSION}}

# Install dependencies
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
  software-properties-common \
  build-essential \
  curl \
  ffmpeg \
  git \
  vim \
  nano \
  rsync \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

  RUN add-apt-repository ppa:deadsnakes/ppa
  RUN apt-get update && apt-get install -y python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-dev \
      python${PYTHON_VERSION}-distutils 

# Set python aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Install pip (we need the latest version not the standard Ubuntu version, to
# support modern wheels)
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py

WORKDIR /home/meltingpot

COPY . .

RUN pip install setuptools && \
    SYSTEM_VERSION_COMPAT=0 pip install dmlab2d && \ 
    pip install -e . && \
    bash ray_patch.sh

CMD ["/bin/bash"]
