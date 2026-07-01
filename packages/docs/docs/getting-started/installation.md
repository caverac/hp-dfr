---
sidebar_position: 1
---

# Installation

This guide covers how to set up the Goal-Oriented DFR environment.

## Prerequisites

- **Python** 3.11, 3.12, or 3.13
- **Node.js** >= 22 (for documentation)
- **Yarn** >= 4 (for monorepo management)
- **uv** (recommended Python package manager)

:::caution Platform Support
Deep learning backends have limited platform support:
- **Apple Silicon (M1/M2/M3)**: All backends supported
- **Intel Mac (x86_64)**: Only PyTorch 2.1.x with Python 3.11
- **Linux**: All backends supported
- **Windows**: TensorFlow and PyTorch supported
:::

## Clone the Repository

```bash
git clone https://github.com/caverac/hp-dfr.git
cd hp-dfr
```

## Install Dependencies

### Node.js Dependencies (Documentation)

```bash
# Enable Corepack for Yarn 4
corepack enable

# Install all workspace dependencies
yarn install
```

### Python Dependencies (Experiments)

We recommend using [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

#### Standard Installation (Apple Silicon / Linux)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment at project root
uv venv --python 3.13
source .venv/bin/activate

# Install with your preferred backend
uv sync --extra pytorch    # PyTorch
uv sync --extra tensorflow # TensorFlow
uv sync --extra all        # All backends
```

#### Intel Mac Installation

Intel Macs require Python 3.11 and older PyTorch versions:

```bash
# Create virtual environment with Python 3.11
uv venv --python 3.11
source .venv/bin/activate

# Install base dependencies
uv sync

# Install PyTorch (only supported backend on Intel Mac)
uv pip install "torch>=2.0.0,<2.2.0"
```

:::warning Intel Mac Limitations
- **TensorFlow**: No longer supports Intel Macs (dropped in TF 2.16+)
- **JAX**: No longer supports Intel Macs (dropped in JAX 0.4.20+)
- **PyTorch**: Only versions 2.0.x-2.1.x support Intel Mac, and only with Python 3.11
:::

### Alternative: pip Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e "packages/experiments[dev,pytorch]"
```

## Backend-Specific Installation

The experiments package supports multiple deep learning backends. Install based on your platform:

### PyTorch (Recommended)

```bash
# Apple Silicon / Linux / Windows
uv sync --extra pytorch

# Intel Mac (Python 3.11 only)
uv pip install "torch>=2.0.0,<2.2.0"
```

### TensorFlow

```bash
# Apple Silicon / Linux / Windows (NOT Intel Mac)
uv sync --extra tensorflow
```

### JAX

```bash
# Linux (CPU)
uv sync --extra jax

# Linux (GPU - CUDA 12)
uv pip install "jax[cuda12]"
```

:::note JAX on macOS
JAX only supports Apple Silicon Macs (M1/M2/M3), not Intel Macs.
:::

## Verify Installation

### Check CLI Installation

```bash
hp-dfr --version
hp-dfr --help
```

### Check Backend Availability

```bash
hp-dfr backends
```

This displays a table showing which backends are installed and their versions.

### Run a Test Experiment

```bash
# With PyTorch backend
hp-dfr pinns run --problem sine --backend pytorch --epochs 100

# With TensorFlow backend (not Intel Mac)
hp-dfr pinns run --problem sine --backend tensorflow --epochs 100
```

## Running the Documentation Site

```bash
# From repository root
yarn docs:dev

# Opens at http://localhost:3000
```

## Docker (Alternative)

For reproducible environments or to avoid platform compatibility issues:

```bash
cd packages/experiments

# Build the image
docker build -t hp-dfr .

# Run experiments
docker run -it hp-dfr hp-dfr pinns run --problem sine --backend pytorch
```

## Troubleshooting

### Intel Mac: No Compatible Backend

**Problem**: TensorFlow, JAX, and recent PyTorch versions don't support Intel Macs.

**Solution**: Use Python 3.11 with PyTorch 2.1.x:

```bash
rm -rf .venv
uv venv --python 3.11
source .venv/bin/activate
uv sync
uv pip install "torch>=2.0.0,<2.2.0"
```

### NumPy 2.x Incompatibility

**Problem**: `A module compiled using NumPy 1.x cannot run in NumPy 2.x`

**Solution**: This occurs when using older PyTorch with NumPy 2.x. The project pins NumPy < 2.0 for compatibility. If you encounter this:

```bash
uv pip install "numpy>=1.24.0,<2.0.0"
```

### Python Version Too New

**Problem**: `no wheels with a matching Python version tag (e.g., cp314)`

**Solution**: Deep learning frameworks need time to support new Python versions. Use a supported version:

```bash
# Check your Python version
python --version

# Create venv with supported version
uv venv --python 3.13  # or 3.12, 3.11
source .venv/bin/activate
```

### macOS Platform Detection Issues

**Problem**: uv reports unusual platform like `macosx_26_0_x86_64`

**Solution**: This can happen with beta macOS versions. Try:

```bash
# Force specific Python version
uv venv --python 3.11
```

### CUDA Not Found (JAX/PyTorch GPU)

**Problem**: GPU not detected despite CUDA being installed.

**Solution**: Ensure CUDA is properly installed and environment is configured:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### TensorFlow GPU Issues

Check GPU availability:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### Memory Issues

For large experiments, limit GPU memory:

```python
# TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# JAX
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# PyTorch
import torch
torch.cuda.set_per_process_memory_fraction(0.8)
```

### ModuleNotFoundError for Backend

**Problem**: `ModuleNotFoundError: No module named 'torch'` (or keras, jax)

**Solution**: Install the appropriate backend extra:

```bash
uv sync --extra pytorch    # For PyTorch
uv sync --extra tensorflow # For TensorFlow
uv sync --extra jax        # For JAX
```

Or install directly:

```bash
uv pip install torch       # PyTorch
uv pip install tensorflow  # TensorFlow
uv pip install jax jaxlib  # JAX
```

## Platform Compatibility Matrix

| Platform | Python | TensorFlow | PyTorch | JAX |
|----------|--------|------------|---------|-----|
| Apple Silicon (M1/M2/M3) | 3.11-3.13 | Yes | Yes | Yes |
| Intel Mac (x86_64) | 3.11 only | No | 2.0-2.1 only | No |
| Linux x86_64 | 3.11-3.13 | Yes | Yes | Yes |
| Linux ARM64 | 3.11-3.13 | Yes | Yes | Yes |
| Windows x64 | 3.11-3.13 | Yes | Yes | No |

## Next Steps

- [Quick Start Guide](/docs/getting-started/quick-start) - Run your first experiment
- [API Reference](/docs/api/tensorflow) - Detailed API documentation
