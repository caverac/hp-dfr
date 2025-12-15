# hp-DFR Experiment Container
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install uv

# Copy package files
COPY pyproject.toml README.md ./
COPY src/ src/

# Install package with TensorFlow backend (default)
RUN uv pip install --system -e ".[tensorflow]"

# Default command
CMD ["hp-dfr", "--help"]

# Example usage:
# docker build -t hp-dfr .
# docker run hp-dfr hp-dfr run --method dfr --problem sine
