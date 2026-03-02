---
title: OmniBench OpenEnv Environment (Submission-Ready)
emoji: 🥁
colorFrom: blue
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - omnibench
  - submission-ready
---

# OmniBench OpenEnv Environment (Windows) — Submission Ready

Este repo contiene un **OpenEnv environment server** preparado para **OmniBench** y verificado con **smoke test 7/7 PASS**.

✅ Dominios que pasan el smoke:
- `finance`
- `agent_safety`
- `healthcare`
- `web`
- `research`
- `coding`
- `computer_use`

🔎 Aserciones importantes del smoke:
- En `computer_use` se usan **clicks por IDs**: `settings_button → dark_mode_toggle`
- En `research` la respuesta esperada es **exactamente**: `OB-Score`

> Nota: Este README reemplaza al anterior que describía un entorno tipo “echo back messages”. :contentReference[oaicite:1]{index=1}

---

## Requisitos (Windows)

- **Docker Desktop** (WSL2 recomendado)
- **Python** (3.10+ recomendado)
- **uv** instalado (se usa `uv run --project . ...`)
- Git

---

## Quick Start (lo más fácil)

### Opción A — 1 comando: Docker + Smoke (recomendado)

Este repo incluye un script de “1 comando” para:
1) build de la imagen  
2) run con port mapping **host 8003 → container 8000**  
3) correr smoke 7/7 contra `http://127.0.0.1:8003`

```powershell
.\scripts\smoke_docker.ps1 -HostPort 8003
