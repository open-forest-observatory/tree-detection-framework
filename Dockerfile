# Stage 1: builder
# Build tools, compilers, and poetry are only needed here.
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git build-essential curl \
    libgl1 libglib2.0-0 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app
COPY . /app

# Install only production dependencies (skip dev group and the root package)
# --no-root avoids baking in the source package so it can be mounted at runtime
RUN /root/.local/bin/poetry config virtualenvs.create false && \
    /root/.local/bin/poetry install --without dev --no-root

# Install detectron2 from source with --no-build-isolation so the build
# subprocess can see torch already installed above by poetry
RUN pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git

# Stage 2: runtime 
# No compilers, no build tools, no poetry — only what's needed to run.
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# libgl1 and libglib2.0-0 are runtime deps for cv2/albumentations
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    curl libgl1 libglib2.0-0 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and entry-point scripts from builder
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Source code is NOT baked in — mount the repo at /app at runtime.
# PYTHONPATH lets Python find tree_detection_framework from the mounted volume.
WORKDIR /app
ENV PYTHONPATH=/app

# Download the detectree2 weights
RUN mkdir -p /app/checkpoints/detectree2 && \
    curl -L -o /app/checkpoints/detectree2/230103_randresize_full.pth \
    https://zenodo.org/records/10522461/files/230103_randresize_full.pth

# Run the detector script
CMD python /app/tree_detection_framework/entrypoints/generate_predictions.py