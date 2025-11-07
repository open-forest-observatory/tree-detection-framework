# Base python image
FROM python:3.10.19-slim

# Install poetry
RUN apt-get update && apt-get install -y curl
RUN curl -sSL https://install.python-poetry.org | python3 -

# Install the module dependencies with poetry without creating a virtual environment
RUN poetry config virtualenvs.create false && /root/.local/bin/poetry install

RUN python -c "import tree_detection_framework; print(tree_detection_framework)"

# Install git
RUN  apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
# Install deepforest dependencies
RUN pip install git+https://github.com/facebookresearch/detectron2.git

# Download the weights
RUN mkdir /app/checkpoints/detectree2 -p && cd /app/checkpoints/detectree2 && curl https://zenodo.org/records/10522461/files/230103_randresize_full.pth

# Note that it's worth trying to install SAM as well in the same container but I'm not sure it will work

# Set the container workdir
WORKDIR /app
# Copy files from current directory into /app
COPY . /app