# CUDA base image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
# Install Python, git, gcc/g++, and curl for downloading and compiling
# Also install libgl1 and libglib2.0-0 for cv2/albumentations dependency
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*
# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set the container workdir
WORKDIR /app
# Copy files from current directory into /app
COPY . /app

# Install the module dependencies with poetry without creating a virtual environment
RUN /root/.local/bin/poetry config virtualenvs.create false && /root/.local/bin/poetry install

# Install deepforest dependencies
#RUN pip install git+https://github.com/facebookresearch/detectron2.git

# Download the weights
#RUN mkdir /app/checkpoints/detectree2 -p && cd /app/checkpoints/detectree2 && curl https://zenodo.org/records/10522461/files/230103_randresize_full.pth

# Note that it's worth trying to install SAM as well in the same container but I'm not sure it will work

# Run the detector script
CMD python /app/tree_detection_framework/entrypoints/generate_predictions.py