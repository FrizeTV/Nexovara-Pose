#!/usr/bin/env bash
# Nexovara Pose: автоматическая установка для Linux / macOS (всё в папку проекта)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,14) else 1)'; then PY="$c"; break; fi
  done
fi
[ -n "$PY" ] || { echo "Нужен Python 3.10-3.14"; exit 1; }
# Qt на Linux может потребовать системные библиотеки: sudo apt install libxcb-cursor0 libegl1
exec "$PY" -I tools/setup.py "$@"
