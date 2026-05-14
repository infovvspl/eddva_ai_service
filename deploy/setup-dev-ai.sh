#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ONE-TIME DEV ENVIRONMENT SETUP — AI Server (PROD-AI / Instance 2)
# Run as: sudo bash setup-dev-ai.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "═══════════════════════════════════════════════════════"
echo "  Setting up DEV AI Service on PROD-AI server"
echo "═══════════════════════════════════════════════════════"

# ── 1. Create directories ────────────────────────────────────────────────────
mkdir -p /home/ubuntu/eddva_ai_service_dev
mkdir -p /home/ubuntu/logs
chown -R ubuntu:ubuntu /home/ubuntu/eddva_ai_service_dev
chown -R ubuntu:ubuntu /home/ubuntu/logs

# ── 2. Open port 8001 in firewall (dev AI port) ──────────────────────────────
ufw allow 8001/tcp
echo "Port 8001 opened."

# ── 2b. Nginx — dev AI domain (dev-ai.eddva.in → port 8001) ─────────────────
cat > /etc/nginx/sites-available/eddva-dev-ai << 'NGINX'
upstream ai_backend_dev {
    server 127.0.0.1:8001;
    keepalive 32;
}

server {
    listen 80;
    server_name dev-ai.eddva.in;

    client_max_body_size 2G;
    proxy_read_timeout 300s;
    proxy_connect_timeout 300s;
    proxy_send_timeout 300s;

    location / {
        proxy_pass http://ai_backend_dev;
        proxy_http_version 1.1;
        proxy_set_header Connection '';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
NGINX

ln -sf /etc/nginx/sites-available/eddva-dev-ai /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
echo "Nginx dev-ai.eddva.in config active."

# ── 3. Clone dev branch ──────────────────────────────────────────────────────
su - ubuntu -c "
  if [ ! -d /home/ubuntu/eddva_ai_service_dev/.git ]; then
    git clone -b dev https://github.com/infovvspl/AI_Study.git /home/ubuntu/eddva_ai_service_dev
    echo 'Dev AI repo cloned.'
  else
    echo 'Dev AI repo already exists, skipping clone.'
  fi
"

# ── 4. Create Python venv ────────────────────────────────────────────────────
su - ubuntu -c "
  cd /home/ubuntu/eddva_ai_service_dev
  if [ ! -d venv ]; then
    python3.11 -m venv venv
    echo 'venv created.'
  fi
  venv/bin/pip install --upgrade pip --quiet
  venv/bin/pip install -r requirements.txt --quiet
  echo 'Dependencies installed.'
"

# ── 5. Create placeholder .env (CI/CD will overwrite on first push) ──────────
cat > /home/ubuntu/eddva_ai_service_dev/.env << 'EOF'
# This file is overwritten by GitHub Actions on every dev deploy.
# Fill in manually only for first-time test before CI/CD is set up.
GROQ_API_KEY=
GEMINI_API_KEY=
SERPAPI_KEY=
DJANGO_SECRET_KEY=change-me-dev-secret-key
ALLOWED_HOSTS=localhost,127.0.0.1
AI_API_KEY=dev-api-key
WHISPER_MODEL=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
DJANGO_DEBUG=true
DB_ENGINE=django.db.backends.sqlite3
DB_NAME=/home/ubuntu/eddva_ai_service_dev/db_dev.sqlite3
DB_USER=
DB_PASSWORD=
DB_HOST=
DB_PORT=
EOF
chown ubuntu:ubuntu /home/ubuntu/eddva_ai_service_dev/.env

# ── 6. Run migrations with placeholder .env ──────────────────────────────────
su - ubuntu -c "
  cd /home/ubuntu/eddva_ai_service_dev
  venv/bin/python manage.py migrate --noinput 2>/dev/null || echo 'Migration skipped (env not configured yet).'
"

# ── 7. Start PM2 dev AI process on port 8001 ─────────────────────────────────
su - ubuntu -c "
  if pm2 list | grep -q ai-service-dev; then
    echo 'ai-service-dev already running in PM2, skipping start.'
  else
    pm2 start \
      '/home/ubuntu/eddva_ai_service_dev/venv/bin/gunicorn ai_study_project.wsgi:application --bind 0.0.0.0:8001 --workers 2 --timeout 120' \
      --name ai-service-dev \
      --interpreter none \
      --log /home/ubuntu/logs/ai-dev.log
    pm2 save
    echo 'ai-service-dev started on port 8001.'
  fi
"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  DEV AI Server setup complete!"
echo ""
echo "  Running:"
echo "  PROD  ai-service     → port 8000"
echo "  DEV   ai-service-dev → port 8001"
echo ""
echo "  NEXT STEPS:"
echo "  1. Add DNS A record:  dev-ai.eddva.in → this server IP"
echo "  2. After DNS propagates, run SSL:"
echo "     certbot --nginx -d dev-ai.eddva.in"
echo "  3. Add GitHub Secrets to AI repo (DEV_* secrets)"
echo "  4. Push to dev branch → CI/CD will deploy automatically"
echo "  5. Check logs: pm2 logs ai-service-dev"
echo "═══════════════════════════════════════════════════════"
