# Choose a specific Python base image (slim versions are smaller)
FROM python:3.10-slim

LABEL maintainer="Your Name <you@example.com>"
LABEL description="Docker image for pyATS MCP Server interacting via stdio"

# Install system dependencies required by pyATS, SSH, and your script
# Combine update, install, and cleanup in one layer to optimize image size
RUN echo "==> Installing System Dependencies..." \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        openssh-client \
        dos2unix \
        build-essential \
        libssl-dev \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        git \
    # Clean up apt cache to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Upgrade pip and install Python dependencies
# Breaking down the installation process to handle problematic packages first
RUN echo "==> Upgrading pip and installing build tools..." \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

# Install the problematic packages first
RUN echo "==> Installing problematic dependencies separately..." \
    && pip install --no-cache-dir \
        backports.ssl \
        backports.ssl-match-hostname \
        tftpy \
        ncclient \
        f5-icontrol-rest

# Then install the main packages
RUN echo "==> Installing pyATS and other Python packages..." \
    && pip install --no-cache-dir \
        pydantic \
        python-dotenv \
        fastmcp \
        pyats[full]==25.2.0

# Copy your application code into the container's working directory
COPY pyats_mcp_server.py .

# Optional: If you have other files needed by the script (e.g., commands.json), copy them too
# COPY commands.json .

# Define the entrypoint to run your server script
ENTRYPOINT ["python", "pyats_mcp_server.py"]

# For the default continuous mode, no CMD arguments are needed
CMD []