FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Streamlit config — disable telemetry, set port
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONIOENCODING=utf-8

EXPOSE 7860

# Verify model file was included (detects unresolved LFS pointers at build time)
RUN echo "=== Model file check ===" && \
    ls -lh models/ && \
    [ $(stat -c%s models/best_model.keras 2>/dev/null || echo 0) -gt 100000 ] && \
    echo "Model file OK" || echo "WARNING: model file may be an LFS pointer or missing"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
