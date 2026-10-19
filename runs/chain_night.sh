#!/bin/bash
cd "$(dirname "$0")/.."
scripts/run_sweeps.sh 13 "" teacher classical dagger_v1 >> runs/sweeps/sweeps.log 2>&1
scripts/night_2026-10-17.sh 13 > runs/night_2026-10-17.log 2>&1
