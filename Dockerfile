FROM python:3.10-slim

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

# Expose ports for both services (documentation only, actual exposure in compose)
EXPOSE 8000
EXPOSE 8501

# The CMD is handled by docker-compose, but we can set a default
CMD ["uv", "run", "uvicorn", "src.nps_latam.api:app", "--host", "0.0.0.0", "--port", "8000"]
