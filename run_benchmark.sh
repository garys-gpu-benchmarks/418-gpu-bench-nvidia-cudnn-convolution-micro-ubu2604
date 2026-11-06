#!/usr/bin/env bash
# File: run_benchmark.sh
# Description: Builds src/cudnn_conv.
# Execution: bash run_benchmark.sh [--smoke|--baseline|--extended]
# Options: --smoke, --baseline, --extended, --config, --help
# Requirements: bash, python3, repository-local .venv
# Dependencies: common.sh, collect_workload.py
# License: Apache-2.0
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_NAME="$(basename "${REPO_ROOT}")"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source scripts/lib/common.sh
capture_benchmark_run_command_submitted "$@"

usage() {
  cat <<'USAGE'
Usage: bash run_benchmark.sh [OPTIONS]

Workload Name:          cuDNN Convolution Microbenchmark.
Main Goal:              Measure cuDNN convolution solver speed.
Validation Objective:   Validates cuDNN convolution timing and TFLOPS from bin/cudnn_conv. Requires
                        a discoverable cudnn.h and libcudnn.
Command Tools Executed: Compile cudnn_conv.cu with nvcc -lcudnn; run bin/cudnn_conv twice via
                        collect_cudnn_conv.py.

Profiles (default: --smoke):
  --smoke                       Run smoke profile
  --baseline                    Run baseline profile
  --extended                    Run extended profile

Parameters (override the profile's value):
  --dtype <value>               Tensor, element, or storage data type (smoke/baseline/extended: FP16)
  --conv-direction <value>      Shared workload parameter (smoke/baseline/extended: all)
  --batch-size <n>              Number of items processed together (smoke 1, baseline 32, extended 32)
  --input-channels <value>      Shared workload parameter (smoke 8, baseline 3, extended 3)
  --input-height <value>        Shared workload parameter (smoke 16, baseline 224, extended 224)
  --input-width <value>         Shared workload parameter (smoke 16, baseline 224, extended 224)
  --output-channels <value>     Shared workload parameter (smoke 8, baseline 64, extended 64)
  --kernel-height <value>       Shared workload parameter (smoke/baseline/extended: 3)
  --kernel-width <value>        Shared workload parameter (smoke/baseline/extended: 3)
  --stride-h <value>            Shared workload parameter (smoke/baseline/extended: 1)
  --stride-w <value>            Shared workload parameter (smoke/baseline/extended: 1)
  --group-count <value>         Shared workload parameter (smoke/baseline/extended: 1)
  --search-strategy <value>     Shared workload parameter (smoke/baseline/extended: find)
  --warmup-iters <n>            Untimed warm-up repetitions before measurement (smoke 1, baseline 5, extended 10)
  --num-iterations <n>          Number of timed repetitions (smoke 2, baseline 57600, extended 194000)

General options:
  --config <file>               Config file (default: config/benchmark_config.yaml)
  --quiet                       Suppress nonessential stdout
  --log-level <level>           ERROR|WARN|INFO|DEBUG (default: INFO; DEBUG also prints [RUN] commands)
  --specification               Print this workload's specification and exit
  --help                        Show this help and exit

Examples:
  bash run_benchmark.sh --smoke
  bash run_benchmark.sh --baseline
  bash run_benchmark.sh --extended
  bash run_benchmark.sh --baseline --batch-size 16
USAGE
}

for _help_arg in "$@"; do
  case "${_help_arg}" in
    --help) usage; exit 0 ;;
    --specification)
      python3 scripts/print_benchmark_definition.py
      exit 0
      ;;
  esac
done

run_profiles_in_order "${REPO_ROOT}/run_benchmark.sh" "$@"

PROFILE="smoke"
CONFIG_FILE="config/benchmark_config.yaml"
QUIET=0
LOG_LEVEL_VALUE="INFO"
DTYPE=""
CONV_DIRECTION=""
BATCH_SIZE=""
INPUT_CHANNELS=""
INPUT_HEIGHT=""
INPUT_WIDTH=""
OUTPUT_CHANNELS=""
KERNEL_HEIGHT=""
KERNEL_WIDTH=""
STRIDE_H=""
STRIDE_W=""
GROUP_COUNT=""
SEARCH_STRATEGY=""
WARMUP_ITERS=""
NUM_ITERATIONS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) PROFILE="smoke"; shift ;;
    --baseline) PROFILE="baseline"; shift ;;
    --extended) PROFILE="extended"; shift ;;
    --config) CONFIG_FILE="${2:-}"; shift 2 ;;
    --validate) shift ;;  # ignored; result validation always runs
    --quiet) QUIET=1; shift ;;
    --log-level) LOG_LEVEL_VALUE="${2:-}"; shift 2 ;;
    --dtype) DTYPE="${2:-}"; shift 2 ;;
    --conv-direction) CONV_DIRECTION="${2:-}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
    --input-channels) INPUT_CHANNELS="${2:-}"; shift 2 ;;
    --input-height) INPUT_HEIGHT="${2:-}"; shift 2 ;;
    --input-width) INPUT_WIDTH="${2:-}"; shift 2 ;;
    --output-channels) OUTPUT_CHANNELS="${2:-}"; shift 2 ;;
    --kernel-height) KERNEL_HEIGHT="${2:-}"; shift 2 ;;
    --kernel-width) KERNEL_WIDTH="${2:-}"; shift 2 ;;
    --stride-h) STRIDE_H="${2:-}"; shift 2 ;;
    --stride-w) STRIDE_W="${2:-}"; shift 2 ;;
    --group-count) GROUP_COUNT="${2:-}"; shift 2 ;;
    --search-strategy) SEARCH_STRATEGY="${2:-}"; shift 2 ;;
    --warmup-iters) WARMUP_ITERS="${2:-}"; shift 2 ;;
    --num-iterations) NUM_ITERATIONS="${2:-}"; shift 2 ;;
    --help|--specification) shift ;;
    *) die "Unknown option: $1" ;;
  esac
done


set_log_level "${LOG_LEVEL_VALUE}"
begin_benchmark_run "${PROFILE}"
bash scripts/ensure_setup.sh
mark_benchmark_measure_start
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
[[ -x "${PYTHON_BIN}" ]] || die "Missing ${PYTHON_BIN}; run bash setup.sh --assume-yes first."

yaml_get() {
  "${PYTHON_BIN}" - "${CONFIG_FILE}" "${PROFILE}" "$1" <<'PY'
import sys
from pathlib import Path
import yaml
key = sys.argv[3]
profile = sys.argv[2]
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
sweep = cfg.get("sweep") or {}
raw_value = sweep.get(key, "")
if isinstance(raw_value, dict):
    raw_value = raw_value.get(profile, next(iter(raw_value.values()), ""))
if isinstance(raw_value, list):
    raw_value = ",".join(str(item) for item in raw_value)
print("" if raw_value is None else raw_value)
PY
}

DTYPE="${DTYPE:-$(yaml_get dtype)}"
CONV_DIRECTION="${CONV_DIRECTION:-$(yaml_get conv_direction)}"
BATCH_SIZE="${BATCH_SIZE:-$(yaml_get batch_size)}"
INPUT_CHANNELS="${INPUT_CHANNELS:-$(yaml_get input_channels)}"
INPUT_HEIGHT="${INPUT_HEIGHT:-$(yaml_get input_height)}"
INPUT_WIDTH="${INPUT_WIDTH:-$(yaml_get input_width)}"
OUTPUT_CHANNELS="${OUTPUT_CHANNELS:-$(yaml_get output_channels)}"
KERNEL_HEIGHT="${KERNEL_HEIGHT:-$(yaml_get kernel_height)}"
KERNEL_WIDTH="${KERNEL_WIDTH:-$(yaml_get kernel_width)}"
STRIDE_H="${STRIDE_H:-$(yaml_get stride_h)}"
STRIDE_W="${STRIDE_W:-$(yaml_get stride_w)}"
GROUP_COUNT="${GROUP_COUNT:-$(yaml_get group_count)}"
SEARCH_STRATEGY="${SEARCH_STRATEGY:-$(yaml_get search_strategy)}"
WARMUP_ITERS="${WARMUP_ITERS:-$(yaml_get warmup_iters)}"
NUM_ITERATIONS="${NUM_ITERATIONS:-$(yaml_get num_iterations)}"
OUTPUT_FORMAT="$(yaml_get output_format)"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-csv}"

BENCHMARK_RUN_COMMAND="bash run_benchmark.sh --${PROFILE}"
[[ -n "${DTYPE}" ]] && BENCHMARK_RUN_COMMAND+=" --dtype $(printf %q "${DTYPE}")"
[[ -n "${CONV_DIRECTION}" ]] && BENCHMARK_RUN_COMMAND+=" --conv-direction $(printf %q "${CONV_DIRECTION}")"
[[ -n "${BATCH_SIZE}" ]] && BENCHMARK_RUN_COMMAND+=" --batch-size $(printf %q "${BATCH_SIZE}")"
[[ -n "${INPUT_CHANNELS}" ]] && BENCHMARK_RUN_COMMAND+=" --input-channels $(printf %q "${INPUT_CHANNELS}")"
[[ -n "${INPUT_HEIGHT}" ]] && BENCHMARK_RUN_COMMAND+=" --input-height $(printf %q "${INPUT_HEIGHT}")"
[[ -n "${INPUT_WIDTH}" ]] && BENCHMARK_RUN_COMMAND+=" --input-width $(printf %q "${INPUT_WIDTH}")"
[[ -n "${OUTPUT_CHANNELS}" ]] && BENCHMARK_RUN_COMMAND+=" --output-channels $(printf %q "${OUTPUT_CHANNELS}")"
[[ -n "${KERNEL_HEIGHT}" ]] && BENCHMARK_RUN_COMMAND+=" --kernel-height $(printf %q "${KERNEL_HEIGHT}")"
[[ -n "${KERNEL_WIDTH}" ]] && BENCHMARK_RUN_COMMAND+=" --kernel-width $(printf %q "${KERNEL_WIDTH}")"
[[ -n "${STRIDE_H}" ]] && BENCHMARK_RUN_COMMAND+=" --stride-h $(printf %q "${STRIDE_H}")"
[[ -n "${STRIDE_W}" ]] && BENCHMARK_RUN_COMMAND+=" --stride-w $(printf %q "${STRIDE_W}")"
[[ -n "${GROUP_COUNT}" ]] && BENCHMARK_RUN_COMMAND+=" --group-count $(printf %q "${GROUP_COUNT}")"
[[ -n "${SEARCH_STRATEGY}" ]] && BENCHMARK_RUN_COMMAND+=" --search-strategy $(printf %q "${SEARCH_STRATEGY}")"
[[ -n "${WARMUP_ITERS}" ]] && BENCHMARK_RUN_COMMAND+=" --warmup-iters $(printf %q "${WARMUP_ITERS}")"
[[ -n "${NUM_ITERATIONS}" ]] && BENCHMARK_RUN_COMMAND+=" --num-iterations $(printf %q "${NUM_ITERATIONS}")"
set_benchmark_run_command "${BENCHMARK_RUN_COMMAND}"

HOSTNAME_VALUE="$(hostname -s 2>/dev/null || hostname)"
RUN_DIR="${REPO_ROOT}/results/raw/$(benchmark_raw_run_dir_name "${REPO_NAME}" "${HOSTNAME_VALUE}")"
mkdir -p "${RUN_DIR}"
RAW_FILE="${RUN_DIR}/raw_output.txt"
set_benchmark_run_dir "${RUN_DIR}"
COMMAND_LOG="${RUN_DIR}/commands_executed.sh"
{ echo "#!/usr/bin/env bash"; echo "set -euo pipefail"; } > "${COMMAND_LOG}"
chmod +x "${COMMAND_LOG}"
printf 'profile=%s\noutput_format=%s\n' "${PROFILE}" "${OUTPUT_FORMAT}" > "${RUN_DIR}/cli_options.env"
export PYTHONUNBUFFERED=1
export PATH="/opt/rocm/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}"
cp run_benchmark.sh "${RUN_DIR}/script.sh"
mask_environment "${RUN_DIR}/env_variables.txt"
: > "${RUN_DIR}/run.log"

run_collection() {
  log_section "Collection"
  local help collect_cmd
  local -a collect_args=(--run-dir "${RUN_DIR}" --raw-file "${RAW_FILE}" --profile "${PROFILE}")
  help="$("${PYTHON_BIN}" scripts/collect_workload.py --help 2>&1 || true)"
  grep -q -- '--config' <<<"${help}" && collect_args+=(--config "${CONFIG_FILE}")
  grep -q -- '--output-format' <<<"${help}" && collect_args+=(--output-format "${OUTPUT_FORMAT}")
  grep -q -- '--dtype' <<<"${help}" && collect_args+=(--dtype "${DTYPE}")
  grep -q -- '--conv-direction' <<<"${help}" && collect_args+=(--conv-direction "${CONV_DIRECTION}")
  grep -q -- '--batch-size' <<<"${help}" && collect_args+=(--batch-size "${BATCH_SIZE}")
  grep -q -- '--input-channels' <<<"${help}" && collect_args+=(--input-channels "${INPUT_CHANNELS}")
  grep -q -- '--input-height' <<<"${help}" && collect_args+=(--input-height "${INPUT_HEIGHT}")
  grep -q -- '--input-width' <<<"${help}" && collect_args+=(--input-width "${INPUT_WIDTH}")
  grep -q -- '--output-channels' <<<"${help}" && collect_args+=(--output-channels "${OUTPUT_CHANNELS}")
  grep -q -- '--kernel-height' <<<"${help}" && collect_args+=(--kernel-height "${KERNEL_HEIGHT}")
  grep -q -- '--kernel-width' <<<"${help}" && collect_args+=(--kernel-width "${KERNEL_WIDTH}")
  grep -q -- '--stride-h' <<<"${help}" && collect_args+=(--stride-h "${STRIDE_H}")
  grep -q -- '--stride-w' <<<"${help}" && collect_args+=(--stride-w "${STRIDE_W}")
  grep -q -- '--group-count' <<<"${help}" && collect_args+=(--group-count "${GROUP_COUNT}")
  grep -q -- '--search-strategy' <<<"${help}" && collect_args+=(--search-strategy "${SEARCH_STRATEGY}")
  grep -q -- '--warmup-iters' <<<"${help}" && collect_args+=(--warmup-iters "${WARMUP_ITERS}")
  grep -q -- '--num-iterations' <<<"${help}" && collect_args+=(--num-iterations "${NUM_ITERATIONS}")

  printf -v collect_cmd '%q ' "${PYTHON_BIN}" scripts/collect_workload.py "${collect_args[@]}"
  log_run "${collect_cmd}"
  echo "${collect_cmd}" >> "${COMMAND_LOG}"
  set +e
  "${PYTHON_BIN}" scripts/collect_workload.py "${collect_args[@]}" 2>&1 | tee -a "${RUN_DIR}/run.log" | hide_run_lines
  command_status="${PIPESTATUS[0]}"
  set -e
  if [[ "${command_status}" -ne 0 ]]; then
    die "collection failed with status ${command_status}"
  fi
}

run_parse() {
  log_section "Parse"
  local parse_cmd
  printf -v parse_cmd '%q ' "${PYTHON_BIN}" scripts/parse_results.py --raw-file "${RAW_FILE}" --db "${REPO_ROOT}/results/benchmark.db" --summary "${REPO_ROOT}/results/summary.json" --definition "${REPO_ROOT}/benchmark_specification.json" --run-dir "${RUN_DIR}"
  log_run "${parse_cmd}"
  echo "${parse_cmd}" >> "${COMMAND_LOG}"
  "${PYTHON_BIN}" scripts/parse_results.py --raw-file "${RAW_FILE}" --db "${REPO_ROOT}/results/benchmark.db" --summary "${REPO_ROOT}/results/summary.json" --definition "${REPO_ROOT}/benchmark_specification.json" --run-dir "${RUN_DIR}" || die "parse failed"
}

run_validate() {
  log_section "Validation"
  local validate_cmd
  printf -v validate_cmd '%q ' "${PYTHON_BIN}" scripts/validate_results.py --db "${REPO_ROOT}/results/benchmark.db" --config "${CONFIG_FILE}"
  log_run "${validate_cmd}"
  echo "${validate_cmd}" >> "${COMMAND_LOG}"
  "${PYTHON_BIN}" scripts/validate_results.py --db "${REPO_ROOT}/results/benchmark.db" --config "${CONFIG_FILE}" || die "validation failed"
}

run_collection
run_parse
run_validate
record_benchmark_stop
capture_journal_warnings "${RUN_DIR}" "${BENCHMARK_START_DATETIME}" "${STOP_TIME}"
bash scripts/collect_hw_sw_info.sh "${RUN_DIR}"
START_EPOCH="$(date -u -d "${BENCHMARK_START_DATETIME}" +%s)"
STOP_EPOCH="$(date -u -d "${STOP_TIME}" +%s)"  # Ledger keeps whole seconds. The summary prints tenths.
ELAPSED="$((STOP_EPOCH - START_EPOCH))"
(( ELAPSED < 0 )) && ELAPSED=0
if [[ "${QUIET}" -eq 0 ]]; then
  bash scripts/print_run_metadata.sh "${REPO_ROOT}" || true
  echo "[INFO] ===== Benchmark Summary ====="
  echo "[INFO] Run ID: $(basename "${RUN_DIR}") | Status: ok | Samples: $(${PYTHON_BIN} -c 'import json; print(json.load(open("results/summary.json"))["sample_count"])') | Profile: ${PROFILE} | Device: $(display_device)"
  print_benchmark_summary_commands
  echo "[INFO] Start time: ${BENCHMARK_START_DATETIME}"
  echo "[INFO] Stop time: ${STOP_TIME}"
  echo "[INFO] Elapsed time: $(benchmark_elapsed_display) sec"
  echo "[INFO] Artifacts: ${RUN_DIR}"
  echo "[INFO] SQLite DB: ${REPO_ROOT}/results/benchmark.db"
  echo "[INFO]"
  "${PYTHON_BIN}" scripts/print_metric_summary.py --definition benchmark_specification.json --summary results/summary.json
  echo "[INFO]"
fi
write_metrics_summary_txt

"${PYTHON_BIN}" scripts/update_runtime_ledger.py \
  --profile "${PROFILE}" --start-datetime "${BENCHMARK_START_DATETIME}" --total-runtime "${ELAPSED}" \
  --exit-code 0 --failure-stage ok --failure-detail "" --raw-run-dir "${RUN_DIR}" \
  --runtime-root "${REPO_ROOT}" \
  --run-benchmark-command-submitted "${BENCHMARK_RUN_COMMAND_SUBMITTED}" \
  --run-benchmark-command-fully-resolved "${BENCHMARK_RUN_COMMAND}" \
  --notes "validated run" \
  || printf '[WARN] Runtime ledger update failed\n' >&2
finish_benchmark_run
