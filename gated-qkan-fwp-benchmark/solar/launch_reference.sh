#!/bin/bash
# Reproduce the authors' GQKAN-QKANFWP on CPU: seeds 0-4, 100 epochs, the
# reference trainer, at most 3 seed processes in parallel with 1 thread each
# (same 3-way rolling window as the reference run.sh), niced.
#
#   nohup bash solar/launch_reference.sh > runs/solar/reference/launch.log 2>&1 &
#   /home/user/venvs/gqkan/bin/python solar/collect_reference.py      # progress / results
#
# Environment overrides: EPOCHS (100), SEEDS ("0 1 2 3 4"), JOBS (3), IMPL (fast).

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "${HERE}")"
PY=${PYTHON:-/home/user/venvs/gqkan/bin/python}
OUT="${PROJECT}/runs/solar/reference"
EPOCHS=${EPOCHS:-100}
JOBS=${JOBS:-3}
IMPL=${IMPL:-fast}
read -r -a SEED_LIST <<< "${SEEDS:-0 1 2 3 4}"

mkdir -p "${OUT}/logs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

sweep_start=$(date +%s)
echo "GQKAN-QKANFWP reference reproduction on CPU: seeds ${SEED_LIST[*]}, ${EPOCHS} epochs, impl=${IMPL}, ${JOBS} parallel"
echo "Started: $(date -u)"

pids=()
for seed in "${SEED_LIST[@]}"; do
    nice -n 19 "${PY}" "${HERE}/reference_cpu.py" \
        --seed "${seed}" --epochs "${EPOCHS}" --impl "${IMPL}" --threads 1 \
        --out_dir "${OUT}/seed${seed}" > "${OUT}/logs/seed${seed}.log" 2>&1 &
    pids+=($!)
    if (( ${#pids[@]} >= JOBS )); then
        wait "${pids[0]}" || echo "  !!! a seed process failed (pid=${pids[0]})"
        pids=("${pids[@]:1}")
    fi
done
for pid in "${pids[@]}"; do
    wait "${pid}" || echo "  !!! a seed process failed (pid=${pid})"
done

echo "SWEEP COMPLETE in $(( $(date +%s) - sweep_start ))s at $(date -u)"
