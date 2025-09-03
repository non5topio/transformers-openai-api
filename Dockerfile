FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY transformers_openai_api/ ./transformers_openai_api/
COPY setup.py .
COPY README.md .
COPY config.example.json .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 apiuser && chown -R apiuser:apiuser /app
USER apiuser

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/v1/engines || exit 1

# Default command
CMD ["gunicorn", "--bind", "0.0.0.0:8001", "--workers", "4", "--timeout", "120", "transformers_openai_api.wsgi:application"]
