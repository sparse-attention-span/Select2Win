#!/bin/bash

# make sure uv is installed, if not:
# curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv erwin
source erwin/bin/activate

MINIMAL=false
if [[ "$*" == *"--minimal"* ]]; then
    MINIMAL=true
fi

# Detect system architecture
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    IS_MACOS_ARM=true
    echo "Detected macOS ARM64 (Apple Silicon)"
else
    IS_MACOS_ARM=false
fi

if [ "$MINIMAL" = true ]; then
    echo "Installing minimal dependencies [Erwin]"

    # Install PyTorch
    if [ "$IS_MACOS_ARM" = true ]; then
        uv pip install torch==2.6.0  # Use latest available for Apple Silicon
    else
        uv pip install torch==2.5.0
        uv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu124.html # CUDA 12.4
    fi

    uv pip install numpy
    uv pip install einops
    uv pip install Cython
    uv pip install setuptools
else
    echo "Installing all dependencies [Erwin + baselines + experiments]"

    # Erwin dependencies
    if [ "$IS_MACOS_ARM" = true ]; then
        uv pip install torch==2.6.0  # Use latest available for Apple Silicon
        # Skip CUDA-specific packages on Apple Silicon
    else
        uv pip install torch==2.5.0 # 2.5 required for torch-scatter
        uv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu124.html # CUDA 12.4

        # CUDA-specific PointTransformer v3 dependencies
        uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html # CUDA 12.4
        uv pip install spconv-cu120
    fi

    uv pip install numpy
    uv pip install einops
    uv pip install Cython
    uv pip install setuptools

    # PointTransformer v3 dependencies
    uv pip install addict
    uv pip install timm

    # MD dependencies
    uv pip install h5py

    # Only install tensorflow on compatible platforms
    if [ "$IS_MACOS_ARM" = false ] || python -c "import sys; sys.exit(int(sys.version_info[:2] >= (3, 13)))"; then
        # cosmology dependencies - skip on Python 3.13+ with macOS ARM64
        uv pip install tensorflow
    fi

    # misc dependencies
    uv pip install wandb
    uv pip install tqdm
    uv pip install matplotlib
fi

# Install C++ balltree implementation
cd balltree
python setup.py install
cd ..