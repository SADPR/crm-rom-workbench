#!/usr/bin/env bash
# Submit the HDM, frozen-PROM, and field-comparison holdout chain.

set -euo pipefail

work_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${work_dir}"

[[ ! -e CleanLaplaceHoldout ]] || {
    echo 'CleanLaplaceHoldout already exists; refusing to overwrite a holdout.' >&2
    exit 1
}
[[ ! -e CleanLaplaceRuns/evaluate/romruns032/point901 ]] || {
    echo 'PROM holdout point901 already exists; refusing to overwrite it.' >&2
    exit 1
}
[[ ! -e CleanLaplaceRuns/evaluate/hromruns032/point901 ]] || {
    echo 'PROM holdout Laplace point901 already exists; refusing to overwrite it.' >&2
    exit 1
}

hdm_submission=$(sbatch --parsable submit_clean_laplace_holdout_hdm.sbatch)
hdm_job=${hdm_submission%%;*}
prom_submission=$(sbatch --parsable --dependency="afterok:${hdm_job}" \
    submit_clean_laplace_holdout_prom.sbatch)
prom_job=${prom_submission%%;*}
comparison_submission=$(sbatch --parsable --dependency="afterok:${prom_job}" \
    submit_clean_laplace_holdout_comparison.sbatch)
comparison_job=${comparison_submission%%;*}

printf 'HDM reference: %s\n' "${hdm_job}"
printf 'Frozen-POD PROM: %s (afterok:%s)\n' "${prom_job}" "${hdm_job}"
printf 'Field comparison: %s (afterok:%s)\n' "${comparison_job}" "${prom_job}"
