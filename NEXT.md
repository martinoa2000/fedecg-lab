# Siguiente paso

Difundir el release 1.0.0: publicar el dashboard estático con `scripts/publish_dashboard.sh` y activar GitHub Pages (Settings > Pages > rama gh-pages).

## Backlog

- [ ] Activar Pages y enlazar la demo del dashboard desde el README
- [ ] Repetir los ajustes federados y de DP con varias semillas (hoy una por ajuste, ruido ~0,003 AUROC)
- [ ] Probar agregación con ruido a nivel de hospital (privacidad de participación), señalada como limitación
- [ ] Vigilar dependencias y CI (uv.lock, versiones de actions) y reproducir con `scripts/reproduce.sh` tras actualizar
- [ ] Contrastar la federación simulada con clientes en procesos separados (Ray), según la limitación documentada
