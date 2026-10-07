# SPEC.md: "The Human How"; Exact technical requirements, environment setup, implementation details, etc.

## Execution Description

## Parameters

| Parameter | CLI Flag | Tested Values | Default | Description |
| --- | --- | --- | --- | --- |
| dtype | `--dtype` | smoke=FP16, baseline=FP16, extended=FP16 | FP16 | From Parameter list; see Execution Description With Parameters. |
| input_source | `--input-source` | smoke=synthetic, baseline=synthetic, extended=synthetic | synthetic | From Parameter list; see Execution Description With Parameters. |
| conv_direction | `--conv-direction` | smoke=all, baseline=all, extended=all | all | From Parameter list; see Execution Description With Parameters. |
| batch_size | `--batch-size` | smoke=1, baseline=32, extended=32 | 32 | From Parameter list; see Execution Description With Parameters. |
| input_channels | `--input-channels` | smoke=8, baseline=3, extended=3 | 3 | From Parameter list; see Execution Description With Parameters. |
| input_height | `--input-height` | smoke=16, baseline=224, extended=224 | 224 | From Parameter list; see Execution Description With Parameters. |
| input_width | `--input-width` | smoke=16, baseline=224, extended=224 | 224 | From Parameter list; see Execution Description With Parameters. |
| output_channels | `--output-channels` | smoke=8, baseline=64, extended=64 | 64 | From Parameter list; see Execution Description With Parameters. |
| kernel_height | `--kernel-height` | smoke=3, baseline=3, extended=3 | 3 | From Parameter list; see Execution Description With Parameters. |
| kernel_width | `--kernel-width` | smoke=3, baseline=3, extended=3 | 3 | From Parameter list; see Execution Description With Parameters. |
| stride_h | `--stride-h` | smoke=1, baseline=1, extended=1 | 1 | From Parameter list; see Execution Description With Parameters. |
| stride_w | `--stride-w` | smoke=1, baseline=1, extended=1 | 1 | From Parameter list; see Execution Description With Parameters. |
| group_count | `--group-count` | smoke=1, baseline=1, extended=1 | 1 | From Parameter list; see Execution Description With Parameters. |
| search_strategy | `--search-strategy` | smoke=find, baseline=find, extended=find | find | From Parameter list; see Execution Description With Parameters. |
| warmup_iters | `--warmup-iters` | smoke=1, baseline=5, extended=10 | 5 | From Parameter list; see Execution Description With Parameters. |
| num_iterations | `--num-iterations` | smoke=2, baseline=85000, extended=250000 | 85000 | From Parameter list; see Execution Description With Parameters. |

## Invocation

```bash
Compile cudnn_conv.cu with nvcc -lcudnn; run bin/cudnn_conv twice via collect_cudnn_conv.py
```

## Raw Output Format

cuDNN convolution CSV from collect_cudnn_conv.py. Forward and both backward directions share one row

check,status,dtype,input_source,conv_direction,batch_size,input_channels,input_height,input_width,output_channels,kernel_height,kernel_width,stride_h,stride_w,group_count,search_strategy,warmup_iters,num_iterations,pad_h,pad_w,dtype_effective,kernel_time_msec,tflops_fwd,tflops_bwd_data,tflops_bwd_weights,solver_search_msec,memory_bandwidth_gb_s,kernel_time_msec_bwd_data,kernel_time_msec_bwd_weights,memory_bandwidth_gb_s_bwd_data,memory_bandwidth_gb_s_bwd_weights,algo_fwd,algo_bwd_data,algo_bwd_weights

## Metrics

- **#1: Convolution kernel execution time, forward** — stored as `kernel_time_msec`.
- **#2: Achieved compute throughput, forward** — stored as `tflops_fwd`.
- **#3: Memory bandwidth, GB/s, forward** — stored as `memory_bandwidth_gb_s`.
- **#4: Solver search time** — stored as `solver_search_msec`.

## Framework

Builds src/cudnn_conv.cu (includes <cudnn.h>) into bin/cudnn_conv and times forward/backward convolution on synthetic tensors. dtype default FP16. conv_direction, batch_size, input_channels, input_height/width, output_channels, kernel_height/width, stride_h/w, group_count, search_strategy, warmup_iters, and num_iterations come from yaml.

## Installation and Execution Summary

Compile src/cudnn_conv.cu with nvcc -lcudnn (requires cudnn.h on the default include path) and run bin/cudnn_conv twice on synthetic tensors. The harness runs FP16 tensor-op convolutions with same padding (kernel/2) and, per direction (fwd, bwd-data, bwd-weights), times cudnnFind algorithm search and the selected algorithm's kernel

## Platform Portability

- **AMD (primary):** ```bash
Compile cudnn_conv.cu with nvcc -lcudnn; run bin/cudnn_conv twice via collect_cudnn_conv.py
```
- **NVIDIA:** Native NVIDIA CUDA workload. Execute on the stated Ubuntu release with the host NVIDIA driver and CUDA userspace. ROCm porting notes do not apply.

## Model Context Protocols

- **Active:** None

## Execution-Loop Validation Contract

EXECUTION CHAIN: `run_benchmark.sh` ➔ raw output ➔ `scripts/parse_results.py` ➔ `results/benchmark.db` ➔ `scripts/validate_results.py`

This benchmark uses a lightweight, SQLite-integrated execution loop for result validation. All validation is performed by `scripts/validate_results.py`.

### Validation script usage

```bash
export BENCHMARK_PYTHON=/usr/bin/python3.13  # optional; select the installed interpreter

# After a live run:
".venv/bin/python" scripts/validate_results.py --db results/benchmark.db

# CI / no-GPU path (seeds fixture and validates it):
".venv/bin/python" scripts/validate_results.py --seed-fixture --quiet

# Override DB path via environment variable:
BENCHMARK_DB=tests/fixtures/benchmark.db \
  ".venv/bin/python" scripts/validate_results.py
```

### Run artifact contract

cuDNN convolution CSV from collect_cudnn_conv.py. Forward and both backward directions share one row

check,status,dtype,input_source,conv_direction,batch_size,input_channels,input_height,input_width,output_channels,kernel_height,kernel_width,stride_h,stride_w,group_count,search_strategy,warmup_iters,num_iterations,pad_h,pad_w,dtype_effective,kernel_time_msec,tflops_fwd,tflops_bwd_data,tflops_bwd_weights,solver_search_msec,memory_bandwidth_gb_s,kernel_time_msec_bwd_data,kernel_time_msec_bwd_weights,memory_bandwidth_gb_s_bwd_data,memory_bandwidth_gb_s_bwd_weights,algo_fwd,algo_bwd_data,algo_bwd_weights

```bash
bash run_benchmark.sh --help
bash run_benchmark.sh --profile smoke --validate
bash run_benchmark.sh --profile baseline --validate
bash run_benchmark.sh --profile extended --validate
```
`run_benchmark.sh --help` prints usage and exits. The harness calls `scripts/ensure_setup.sh` when `.setup_state` is absent.

### Required integrity checks (built into `validate_results.py`)

1. Latest run exists and `runs.status = 'ok'`.
2. `run.error_message` is NULL.
3. `started_at` and `finished_at` are valid ISO-8601 UTC strings.
4. All required aggregate metrics in `runs` are non-NULL and finite.
5. All required aggregate metrics are physically sensible (positive values). Builds src/cudnn_conv.cu (includes <cudnn.h>) into bin/cudnn_conv and times forward/backward convolution on synthetic tensors. dtype default FP16. conv_direction, batch_size, input_channels, input_height/width, output_channels, kernel_height/width, stride_h/w, group_count, search_strategy, warmup_iters, and num_iterations come from yaml.
6. At least 2 sample rows exist for the latest `run_id` (sweep coverage).
7. No sample has `status = 'error'`.
8. Builds src/cudnn_conv.cu (includes <cudnn.h>) into bin/cudnn_conv and times forward/backward convolution on synthetic tensors. dtype default FP16. conv_direction, batch_size, input_channels, input_height/width, output_channels, kernel_height/width, stride_h/w, group_count, search_strategy, warmup_iters, and num_iterations come from yaml.

### Baseline / Threshold configuration (`config/benchmark_config.yaml`)

Expected ranges and gates live in `config/benchmark_config.yaml` under `baselines:` or `thresholds:`. To update them, edit that file — never edit validation code directly.

Threshold key suffixes encode comparison direction when `thresholds:` is present: `_min` → observed value must be ≥ threshold. `_max` → observed value must be ≤ threshold. Informational `baselines:` ranges are not pass/fail gates.
