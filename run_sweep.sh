#!/bin/bash
cd /home/pepebeats/SCL_2.0
export PATH="/home/pepebeats/SCL_2.0/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
python -u scripts/sweep_rvae.py --mode adaptive --max-attempts 15 > sweep_rvae.log 2>&1
