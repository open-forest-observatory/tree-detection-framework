# Base python image
FROM python:3.10.19-slim
# Install git, gcc/g++, and curl
RUN  apt-get update && apt-get install -y git build-essentials curl && rm -rf /var/lib/apt/lists/*
# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set the container workdir
WORKDIR /app
# Copy files from current directory into /app
COPY . /app

# Install the module dependencies with poetry without creating a virtual environment
RUN /root/.local/bin/poetry config virtualenvs.create false && /root/.local/bin/poetry install

RUN python -c "import tree_detection_framework; print(tree_detection_framework)"

# Install deepforest dependencies
RUN pip install git+https://github.com/facebookresearch/detectron2.git

# Download the weights
RUN mkdir /app/checkpoints/detectree2 -p && cd /app/checkpoints/detectree2 && curl https://zenodo.org/records/10522461/files/230103_randresize_full.pth

# Note that it's worth trying to install SAM as well in the same container but I'm not sure it will work
