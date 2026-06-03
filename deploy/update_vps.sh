#!/usr/bin/env bash
# update_vps.sh — Actualizar la app desde GitHub (usar tras cada push)
set -euo pipefail

APP_DIR="/var/www/tesis-cariari"
SERVICE_NAME="guia-sbn"

echo "=== Pull desde GitHub ==="
git -C "$APP_DIR" pull

echo "=== Actualizar dependencias Python ==="
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt" -q

echo "=== Reiniciar servicio ==="
systemctl restart "$SERVICE_NAME"
systemctl status "$SERVICE_NAME" --no-pager

echo "Listo — https://muni-ia.com/sbn/"
