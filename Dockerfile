# ── AI Service (Django + gunicorn) ───────────────────────────────────────────
FROM python:3.11-slim

# System dependencies for OpenCV, PDF processing, EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    ffmpeg \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
# CPU-only torch first (the default PyPI wheel drags in ~4.5 GB of CUDA libs
# that are useless in a CPU container), then everything else.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.prod.txt ./requirements.txt
RUN grep -vE '^(torch|torchvision)\b' requirements.txt > requirements.notorch.txt && \
    pip install --no-cache-dir -r requirements.notorch.txt

# Copy application source
COPY manage.py ./
COPY ai_study_project/ ./ai_study_project/
COPY ai_services/ ./ai_services/

# Runtime dirs: app data + model caches (mount /models as a volume so
# easyocr / whisper weights are not re-downloaded on every container start)
RUN mkdir -p data/uploads /models/easyocr /models/hf
ENV EASYOCR_MODULE_PATH=/models/easyocr \
    HF_HOME=/models/hf \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid 1001 --create-home --home-dir /home/appuser --shell /usr/sbin/nologin appuser
ENV HOME=/home/appuser
RUN chown -R appuser:appgroup /app /models
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/notes/health/', timeout=4)" || exit 1

# Run the Django app via gunicorn — matches the production PM2 setup
# (deploy/ecosystem.config.js). Tune --workers based on EC2 RAM.
CMD ["gunicorn", "ai_study_project.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "3", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]