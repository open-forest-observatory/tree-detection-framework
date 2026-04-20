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

# Install SAM2 from source
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/sam2.git

# SAM3's __init__.py imports torch at module level, so the wheel build needs torch
# available — use --no-build-isolation to share the current environment
RUN pip install --no-build-isolation --no-cache-dir \
    git+https://github.com/facebookresearch/sam3.git \
    decord

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

# Download detectree2 and SAM2 weights; copy locally-saved SAM3 weights
RUN mkdir -p /app/checkpoints && \
    curl -L -o /app/checkpoints/230103_randresize_full.pth \
        https://zenodo.org/records/10522461/files/230103_randresize_full.pth && \
    curl -L -o /app/checkpoints/sam2.1_hiera_large.pt \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

COPY checkpoints/sam3.pt /app/checkpoints/sam3.pt

# Run the detector script
CMD python /app/tree_detection_framework/entrypoints/generate_predictions.py