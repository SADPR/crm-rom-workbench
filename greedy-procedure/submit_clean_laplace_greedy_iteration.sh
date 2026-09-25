#!/usr/bin/env bash
# Submit one restart-safe residual-greedy iteration after the clean 32-state POD.

set -euo pipefail

iteration=${1:?Usage: ./submit_clean_laplace_greedy_iteration.sh ITERATION [CANDIDATE_COUNT]}
candidate_count=${2:-24}

python3 -B clean_laplace_greedy.py screen-init \
    --iteration "${iteration}" --candidate-count "${candidate_count}"

screen_job=$(sbatch --parsable --array="1-${candidate_count}%3" \
    --export="ALL,GREEDY_ITERATION=${iteration}" \
    submit_clean_laplace_greedy_screen.sbatch)
select_job=$(sbatch --parsable --dependency="afterok:${screen_job}" \
    --export="ALL,GREEDY_ITERATION=${iteration}" \
    --output="CleanLaplaceGreedy40/screening/select_${iteration}_%j.out" \
    --error="CleanLaplaceGreedy40/screening/select_${iteration}_%j.err" \
    submit_clean_laplace_greedy_select.sbatch)
hdm_job=$(sbatch --parsable --dependency="afterok:${select_job}" \
    --export="ALL,GREEDY_ITERATION=${iteration}" \
    --output="CleanLaplaceGreedy40/greedy_hdm_${iteration}_%j.out" \
    --error="CleanLaplaceGreedy40/greedy_hdm_${iteration}_%j.err" \
    submit_clean_laplace_greedy_hdm.sbatch)
pod_job=$(sbatch --parsable --dependency="afterok:${hdm_job}" \
    --export="ALL,GREEDY_ITERATION=${iteration}" \
    --output="CleanLaplaceGreedy40/greedy_pod_${iteration}_%j.out" \
    --error="CleanLaplaceGreedy40/greedy_pod_${iteration}_%j.err" \
    submit_clean_laplace_greedy_pod.sbatch)

printf 'Screen array: %s\n' "${screen_job}"
printf 'Selection:    %s (afterok:%s)\n' "${select_job}" "${screen_job}"
printf 'HDM:          %s (afterok:%s)\n' "${hdm_job}" "${select_job}"
printf 'POD:          %s (afterok:%s)\n' "${pod_job}" "${hdm_job}"
