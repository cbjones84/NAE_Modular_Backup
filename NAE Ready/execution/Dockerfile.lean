FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Clone LEAN
WORKDIR /opt
RUN git clone https://github.com/QuantConnect/Lean.git

# Build LEAN
WORKDIR /opt/Lean
RUN dotnet build

# Runtime stage
FROM mcr.microsoft.com/dotnet/runtime:8.0

WORKDIR /app

# Copy LEAN binaries
COPY --from=build /opt/Lean /opt/lean

# Copy execution engine code
COPY execution_engine/ ./execution_engine/
COPY broker_adapters/ ./broker_adapters/

# Install Python dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Set environment variables
ENV PYTHONPATH=/app
ENV LEAN_PATH=/opt/lean
ENV ALGORITHM_PATH=/app/algorithms/nae_signal_consumer

# Run execution manager
CMD ["python3", "-m", "execution_engine.execution_manager"]

