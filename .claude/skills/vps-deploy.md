# vps-deploy — Despliegue en VPS Hostinger

Skill para conectar, instalar y actualizar la Guía SbN Cariari en el VPS de Hostinger bajo `muni-ia.com/sbn`.

## Contexto del proyecto

- **App:** FastAPI sirviendo archivos estáticos de `resultados_refinados/Guia/`
- **VPS:** Hostinger (Ubuntu/Debian), acceso por SSH usuario+contraseña
- **URL destino:** `https://muni-ia.com/sbn/`
- **Puerto interno:** `127.0.0.1:8010` (gunicorn + UvicornWorker)
- **Directorio en VPS:** `/var/www/tesis-cariari`
- **Servicio systemd:** `guia-sbn`
- **Archivos de deploy:** `deploy/` en la raíz del repo

## Datos de conexión SSH

> Pedir al usuario antes de cada sesión de deploy:
> - IP del VPS (visible en panel Hostinger → VPS → detalles)
> - Usuario (generalmente `root`)
> - Contraseña SSH

Conectar:
```bash
ssh root@<IP_VPS>
```

## Primera instalación (desde cero)

```bash
# En el VPS como root:
curl -O https://raw.githubusercontent.com/<usuario>/<repo>/main/deploy/setup_vps.sh
bash setup_vps.sh
```

O manualmente:
```bash
apt-get update && apt-get install -y git python3 python3-venv nginx
git clone https://github.com/<usuario>/<repo>.git /var/www/tesis-cariari
cd /var/www/tesis-cariari
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp deploy/guia-sbn.service /etc/systemd/system/
systemctl enable guia-sbn && systemctl start guia-sbn
```

## Configurar Nginx (subpath /sbn)

Abrir el archivo de configuración del sitio principal:
```bash
nano /etc/nginx/sites-available/muni-ia.com
```

Agregar dentro del bloque `server {}` (ver `deploy/nginx-sbn.conf`):
```nginx
location /sbn {
    return 301 /sbn/;
}
location /sbn/ {
    proxy_pass         http://127.0.0.1:8010/;
    proxy_set_header   Host              $host;
    proxy_set_header   X-Real-IP         $remote_addr;
    proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header   X-Forwarded-Proto $scheme;
    proxy_set_header   X-Forwarded-Prefix /sbn;
}
```

Validar y recargar:
```bash
nginx -t && systemctl reload nginx
```

## Actualizar tras un push a GitHub

```bash
ssh root@<IP_VPS>
bash /var/www/tesis-cariari/deploy/update_vps.sh
```

O manualmente:
```bash
cd /var/www/tesis-cariari
git pull
.venv/bin/pip install -r requirements.txt
systemctl restart guia-sbn
```

## Comandos útiles de mantenimiento

```bash
# Ver estado del servicio
systemctl status guia-sbn

# Ver logs en tiempo real
journalctl -u guia-sbn -f

# Reiniciar servicio
systemctl restart guia-sbn

# Ver logs de nginx
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log

# Verificar que corre en puerto 8010
curl http://127.0.0.1:8010/
```

## Variable de entorno importante

La app usa `ROOT_PATH=/sbn` (configurada en `deploy/guia-sbn.service`) para que FastAPI genere redirects correctos bajo el subpath `/sbn`.

## Estructura de archivos en el VPS

```
/var/www/tesis-cariari/
├── app.py
├── requirements.txt
├── resultados_refinados/
│   └── Guia/
│       └── index.html   ← página principal
├── deploy/
│   ├── guia-sbn.service
│   ├── nginx-sbn.conf
│   ├── setup_vps.sh
│   └── update_vps.sh
└── .venv/
```

## Troubleshooting

| Síntoma | Causa probable | Solución |
|---------|---------------|----------|
| 502 Bad Gateway | gunicorn no corre | `systemctl restart guia-sbn` |
| `/sbn` redirige a raíz incorrecta | `ROOT_PATH` no seteado | Ver variable en `.service` |
| Estáticos 404 | `resultados_refinados/` no clonado | `git pull` o verificar `.gitignore` |
| Cambios no aparecen | Caché nginx | `systemctl reload nginx` |
