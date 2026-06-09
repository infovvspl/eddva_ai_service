# ── AI Service (Django + gunicorn) ───────────────────────────────────────────
FROM python:3.11-slim

# System dependencies for OpenCV, PDF processing, EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.prod.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY manage.py ./
COPY ai_study_project/ ./ai_study_project/
COPY ai_services/ ./ai_services/

# Create data dir (needed by some services at runtime)
RUN mkdir -p data/uploads

# Non-root user
RUN addgroup --gid 1001 --system appgroup && \
    adduser --uid 1001 --system --gid 1001 appuser
RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

# Run the Django app via gunicorn — matches the production PM2 setup
# (deploy/ecosystem.config.js). Tune --workers based on EC2 RAM.
CMD ["gunicorn", "ai_study_project.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "3", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]