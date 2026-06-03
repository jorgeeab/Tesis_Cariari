#!/usr/bin/env bash
# setup_vps.sh — Primera instalación en VPS Hostinger (Ubuntu/Debian)
# Ejecutar como root: bash setup_vps.sh
set -euo pipefail

REPO_URL="https://github.com/jorgeeab/tesis-cariari.git"   # <-- ajustar URL del repo
APP_DIR="/var/www/tesis-cariari"
SERVICE_NAME="guia-sbn"
NGINX_CONF="/etc/nginx/sites-available/muni-ia.com"

echo "=== 1. Dependencias del sistema ==="
apt-get update -qq
apt-get install -y git python3 python3-venv python3-pip nginx

echo "=== 2. Clonar repositorio ==="
if [ -d "$APP_DIR" ]; then
    echo "Directorio ya existe — haciendo pull"
    git -C "$APP_DIR" pull
else
    git clone "$REPO_URL" "$APP_DIR"
fi

echo "=== 3. Entorno virtual y dependencias Python ==="
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip -q
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt" -q

echo "=== 4. Servicio systemd ==="
cp "$APP_DIR/deploy/guia-sbn.service" "/etc/systemd/system/${SERVICE_NAME}.service"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"
echo "Estado del servicio:"
systemctl status "$SERVICE_NAME" --no-pager

echo "=== 5. Nginx — agregar bloque /sbn ==="
echo ""
echo "ACCION MANUAL REQUERIDA:"
echo "Abrir $NGINX_CONF y agregar dentro del bloque server{} el contenido de:"
echo "  $APP_DIR/deploy/nginx-sbn.conf"
echo ""
echo "Luego ejecutar:"
echo "  nginx -t && systemctl reload nginx"
echo ""
echo "=== Instalacion completada ==="
echo "La app corre en: http://127.0.0.1:8010"
echo "Acceso publico (tras nginx): https://muni-ia.com/sbn/"
