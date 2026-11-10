#!/usr/bin/env bash
cd "$(dirname "$0")"
exec .venv/bin/python -I tools/osc_monitor.py "$@"
