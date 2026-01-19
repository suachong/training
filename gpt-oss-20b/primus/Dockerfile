# ARG BASE_IMAGE=rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0
ARG BASE_IMAGE=docker.io/rocm/primus:v25.11
FROM ${BASE_IMAGE}

# Set working directory
WORKDIR /workspace/deps

# Clone Primus repository (remove existing if present in base image)
RUN rm -rf Primus && \
    git clone --recursive https://github.com/AMD-AIG-AIMA/Primus.git && \
    cd Primus && \
    # main branch with evaluation bug fix 
    git checkout 85c51c0da12f7d9b819f944eba9ffeb313795b9a && \
    git submodule update --init --recursive && \
    pip install -r requirements.txt

# Build Megatron C++ extensions (required for dataset helpers)
RUN cd /workspace/deps/Primus/third_party/Megatron-LM && \
    pip install -e . --no-deps

WORKDIR /workspace/code
# Copy the current state of the code inside the image
COPY . .

# Install primus-mllog from local wheel
RUN pip install primus_mllog-0.1.0-py3-none-any.whl