FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-install-project

# Copy project files
COPY . .

# Install the project itself
RUN uv sync --frozen

# Expose ports
EXPOSE 7860
EXPOSE 8000
EXPOSE 5000

# Fix permissions for HF Spaces (runs as arbitrary user)
RUN chmod -R 777 /app/Data /app/mlruns

# Copy and setup entrypoint
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Run all services
CMD ["./entrypoint.sh"]
