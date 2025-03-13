# # ✅ Base image with CUDA 11.8 (matches cu118)
# FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# # ✅ Set non-interactive mode
# ENV DEBIAN_FRONTEND=noninteractive

# # ✅ Ensure CUDA is properly configured
# ENV PATH=/usr/local/cuda/bin:$PATH
# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# ENV CUDA_HOME=/usr/local/cuda

# # ✅ Install system dependencies
# RUN apt update && apt install -y \
#     python3.8 python3.8-venv python3.8-dev \
#     python3-pip \
#     git \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     ninja-build \
#     ffmpeg \
#     libsm6 \
#     libxext6 \
#     && rm -rf /var/lib/apt/lists/*  # ✅ Cleanup to reduce image size

# # ✅ Set Python 3.8 as default
# RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
#     ln -sf /usr/bin/python3.8 /usr/bin/python3

# # ✅ Create working directory
# WORKDIR /workspace

# # ✅ Copy the project files
# COPY . /workspace

# # ✅ Upgrade pip
# RUN python -m pip install --upgrade pip setuptools wheel

# # ✅ Install PyTorch & torchvision for CUDA 11.8 (cu118)
# RUN pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# # ✅ Install mmcv-full from OpenMMLab repository (fixes build issue)
# RUN pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

# # ✅ Install other dependencies
# RUN pip install \
#     numpy==1.23.4 \
#     mmcls==0.25.0 \
#     mmdet==2.28.2 \
#     nuscenes-devkit==1.1.11 \
#     av2==0.2.1 \
#     opencv-python \
#     matplotlib \
#     scipy \
#     pandas \
#     shapely \
#     seaborn \
#     trimesh \
#     pyquaternion \
#     pytorch-sphinx-theme \
#     sphinx \
#     pytest \
#     requests

# # ✅ Install Detectron2 (Official build)
# RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# # ✅ Install the project in development mode
# #RUN python setup.py develop

# # ✅ Compile custom CUDA operators
# #RUN cd mmdet3d/ops && python setup.py build_ext --inplace && \
#  #   cd ../deformattn && python setup.py build install

# # ✅ Set entrypoint to bash for interactive mode
# ENTRYPOINT ["/bin/bash"]


## CUDA 1.6 

# Use a base image with CUDA 11.6 and Python 3.8
# FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# # Set up environment variables
# ENV DEBIAN_FRONTEND=noninteractive \
#     PYTHONUNBUFFERED=1 \
#     LANG=C.UTF-8 \
#     PATH="/opt/conda/bin:$PATH"

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     git \
#     wget \
#     vim \
#     curl \
#     unzip \
#     ffmpeg \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# # Install Miniconda
# WORKDIR /tmp
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh -O miniconda.sh && \
#     bash miniconda.sh -b -p /opt/conda && \
#     rm miniconda.sh && \
#     /opt/conda/bin/conda clean -afy

# # Set up conda environment
# RUN conda create -n mmdetection python=3.8.13 -y
# SHELL ["conda", "run", "-n", "mmdetection", "/bin/bash", "-c"]

# # Install PyTorch and CUDA dependencies
# RUN pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html


# # Install NumPy
# RUN pip install numpy==1.19.5

# # Install OpenMMLab dependencies
# RUN pip install mmcv-full==1.6.0
# RUN pip install mmcls==1.0.0rc1
# RUN pip install mmdet==2.24.0

# # Install NuScenes devkit
# RUN pip install nuscenes-devkit==1.1.9

# # Install Detectron2
# RUN pip install git+https://github.com/facebookresearch/detectron2.git



# # Set default shell to use the conda environment
# RUN echo "conda activate mmdetection" >> ~/.bashrc
# CMD ["/bin/bash"]

# Use a base image with CUDA 12.1 and Ubuntu 20.04
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    PATH="/opt/conda/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    curl \
    unzip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
WORKDIR /tmp
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda clean -afy

# Set up conda environment
RUN conda create -n mmdetection python=3.8.13 -y
SHELL ["conda", "run", "-n", "mmdetection", "/bin/bash", "-c"]

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch and TorchVision for CUDA 12.1 (cu118)
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install NumPy
RUN pip install numpy==1.23.4

# Install OpenMMLab dependencies
RUN pip install mmcv-full==1.6.0
RUN pip install mmcls==0.25.0
RUN pip install mmdet==2.28.2

# Install NuScenes devkit and AV2
RUN pip install nuscenes-devkit==1.1.11 av2==0.2.1

# Install Detectron2
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN git clone https://github.com/facebookresearch/detectron2.git
RUN python -m pip install -e detectron2

# Set default shell to use the conda environment
RUN echo "conda activate mmdetection" >> ~/.bashrc
CMD ["/bin/bash"]

