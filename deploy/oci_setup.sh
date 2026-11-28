#!/usr/bin/env bash
# Run on the Oracle Cloud VM (Ubuntu 22.04 or 24.04, Ampere A1) from the repository folder:
#     bash deploy/oci_setup.sh
# Safe to run again after `git pull` to redeploy.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v docker >/dev/null; then
  echo ">> Installing Docker"
  sudo apt-get update -y
  sudo apt-get install -y docker.io
  sudo systemctl enable --now docker
fi

if ! sudo iptables -C INPUT -p tcp --dport 8501 -j ACCEPT 2>/dev/null; then
  echo ">> Opening port 8501 in the VM firewall (OCI Ubuntu images block it by default)"
  sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8501 -j ACCEPT
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y iptables-persistent
  sudo netfilter-persistent save
fi

echo ">> Building the image (trains the model inside the image)"
sudo docker build -t retail-intelligence .

echo ">> Starting the app"
sudo docker rm -f retail-intelligence 2>/dev/null || true
ENV_ARGS=""; [ -f .env ] && ENV_ARGS="--env-file .env"
sudo docker run -d --name retail-intelligence --restart unless-stopped $ENV_ARGS \
  -p 8501:8501 retail-intelligence

echo ">> Done. Open http://$(curl -s ifconfig.me):8501"
