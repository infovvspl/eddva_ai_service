#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ONE-TIME SETUP — AI Server (Instance 2)
# Ubuntu 22.04 LTS  |  Recommended: t3.large (8 GB RAM) or c5.xlarge
# Installs: Python 3.11, pip, virtualenv, PM2 (via Node), Nginx, system libs
#
# Run as: sudo bash setup-ai-server.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "═══════════════════════════════════════════════════════"
echo "  EDDVA AI Server Setup"
echo "═══════════════════════════════════════════════════════"

# ── 1. System packages ────────────────────────────────────────────────────────
apt-get update -y
apt-get install -y \
  python3.11 python3.11-venv python3.11-dev python3-pip \
  build-essential git nginx ufw \
  # OpenCV & ML system deps
  libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
  # PDF processing
  poppler-utils libpoppler-dev \
  # Audio
  portaudio19-dev libsndfile1 ffmpeg

# ── 2. Node + PM2 (PM2 manages gunicorn process) ─────────────────────────────
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
npm install -g pm2
pm2 startup systemd -u ubuntu --hp /home/ubuntu
env PATH=$PATH:/usr/bin pm2 startup systemd -u ubuntu --hp /home/ubuntu

# ── 3. Firewall — only allow traffic from App Server ─────────────────────────
# IMPORTANT: Replace APP_SERVER_IP with Instance 1's private IP
APP_SERVER_IP="${1:-0.0.0.0/0}"   # pass as first arg: sudo bash setup.sh 10.0.1.X

ufw allow OpenSSH
ufw allow 'Nginx HTTP'
# Allow port 8000 only from App Server private IP
if [ "$APP_SERVER_IP" != "0.0.0.0/0" ]; then
  ufw allow from "$APP_SERVER_IP" to any port 8000
  echo "Port 8000 restricted to App Server IP: $APP_SERVER_IP"
else
  ufw allow 8000
  echo "WARNING: Port 8000 is open to all. Pass App Server IP as arg to restrict."
fi
ufw --force enable

# ── 4. AI service directory ───────────────────────────────────────────────────
mkdir -p /home/ubuntu/ai-service
mkdir -p /home/ubuntu/logs
mkdir -p /home/ubuntu/ai-service/data/uploads
chown -R ubuntu:ubuntu /home/ubuntu/ai-service /home/ubuntu/logs

# ── 5. Python virtual environment ────────────────────────────────────────────
# (Cloned repo + venv setup happens at deploy time)
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  AI Server setup complete!"
echo ""
echo "  NEXT STEPS:"
echo "  1. Copy your .env to /home/ubuntu/ai-service/.env"
echo "  2. Run: git clone <your-repo> /home/ubuntu/ai-service"
echo "  3. Run: cd /home/ubuntu/ai-service && python3.11 -m venv venv"
echo "  4. Run: venv/bin/pip install -r requirements.prod.txt"
echo "  5. Run: pm2 start deploy/ecosystem.config.js"
echo "  6. In NestJS .env: AI_BASE_URL=http://<AI-PRIVATE-IP>:8000"
echo "═══════════════════════════════════════════════════════"
