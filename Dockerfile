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

# Install detectron2 from source with --no-build-isolation so the build
# subprocess can see torch (already installed above by poetry)
RUN pip install --no-build-isolation \
    git+https://github.com/facebookresearch/detectron2.git

# Download the detectree2 weights
RUN mkdir -p /app/checkpoints/detectree2 && \
    curl -L -o /app/checkpoints/detectree2/230103_randresize_full.pth \
    https://zenodo.org/records/10522461/files/230103_randresize_full.pth

# Run the detector script
CMD python /app/tree_detection_framework/entrypoints/generate_predictions.py