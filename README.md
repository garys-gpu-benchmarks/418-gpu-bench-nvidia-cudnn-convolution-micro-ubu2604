# cuDNN Convolution Microbenchmark

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/garys-gpu-benchmarks/418-gpu-bench-nvidia-cudnn-convolution-micro-ubu2604/actions/workflows/ci.yml/badge.svg)](https://github.com/garys-gpu-benchmarks/418-gpu-bench-nvidia-cudnn-convolution-micro-ubu2604/actions/workflows/ci.yml)

Measures cuDNN convolution solver speed. Workload 418 of the NVIDIA Ubuntu 26.04 bundle.

## Quick Start

On an Ubuntu 26.04 host, as root or with sudo:

```bash
git clone https://github.com/garys-gpu-benchmarks/418-gpu-bench-nvidia-cudnn-convolution-micro-ubu2604.git
cd 418-gpu-bench-nvidia-cudnn-convolution-micro-ubu2604
sudo bash setup.sh --assume-yes
bash run_benchmark.sh --smoke
```

`run_benchmark.sh` runs setup by itself if it has not been run yet. Each run prints its metrics when it finishes.

## Profiles and Parameters

| Parameter | Flag | smoke | baseline | extended |
|---|---|---|---|---|
| `dtype` | `--dtype` | FP16 | FP16 | FP16 |
| `conv_direction` | `--conv-direction` | all | all | all |
| `batch_size` | `--batch-size` | 1 | 32 | 32 |
| `input_channels` | `--input-channels` | 8 | 3 | 3 |
| `input_height` | `--input-height` | 16 | 224 | 224 |
| `input_width` | `--input-width` | 16 | 224 | 224 |
| `output_channels` | `--output-channels` | 8 | 64 | 64 |
| `kernel_height` | `--kernel-height` | 3 | 3 | 3 |
| `kernel_width` | `--kernel-width` | 3 | 3 | 3 |
| `stride_h` | `--stride-h` | 1 | 1 | 1 |
| `stride_w` | `--stride-w` | 1 | 1 | 1 |
| `group_count` | `--group-count` | 1 | 1 | 1 |
| `search_strategy` | `--search-strategy` | find | find | find |
| `warmup_iters` | `--warmup-iters` | 1 | 5 | 10 |
| `num_iterations` | `--num-iterations` | 2 | 57600 | 194000 |

Pick a profile with `--smoke` (default), `--baseline` or `--extended`. Any flag in the table overrides that profile's value:

```bash
bash run_benchmark.sh --baseline --batch-size 16
bash run_benchmark.sh --help        # every option
```

Profile values live in `config/benchmark_config.yaml`.

## Metrics

| # | Metric | Key |
|---|---|---|
| 1 | Convolution kernel execution time, forward | `kernel_time_msec` |
| 2 | Achieved convolution throughput, forward, TFLOP/s | `tflops_fwd` |
| 3 | Minimum forward operand traffic, GB/s | `min_operand_traffic_gb_s` |
| 4 | Solver search time, sum of cudnnFind wall time across the directions run | `solver_search_msec` |

### Notes

- The smoke profile is a quick check with reduced sizes and iteration counts. Use `--baseline` or `--extended` for representative numbers.
- cudnn.h must be on the nvcc default include path.
- Solver choice varies by shape and dtype; warm-up is recommended.

## Output

| Path | Contents |
|---|---|
| `results/raw/YYYYMMDD_HHMMSSZ_WORKLOAD-<NNN>_SYSTEM-<hostname>/` | Raw tool output, `metrics_summary.txt`, and `hardware_info.txt`, `software_info.txt`, `errors_info.txt` |
| `results/benchmark.db` | SQLite history of every run |
| `results/summary.json` | Metrics from the most recent run |
| `/var/opt/benchmarks/runtime_ledger.csv` | One row per run, shared by every workload on the host |

Every run is checked for completeness and against any `thresholds:` or `baselines:` in `config/benchmark_config.yaml`. A run that fails the checks exits non-zero.

## Requirements

- Ubuntu 26.04, root or sudo for `setup.sh`
- NVIDIA GPU with CUDA 13.3
- CUDA Runtime, NVCC, cuDNN, Python 3.14. `setup.sh` installs what is missing; Python packages go into a repo-local `.venv`.

## Run the Whole Bundle

This workload is also part of [`bundle-nvidia-ubuntu-2604`](https://github.com/garys-gpu-benchmarks/bundle-nvidia-ubuntu-2604), which holds all 32 NVIDIA Ubuntu 26.04 workloads as submodules:

```bash
git clone --recurse-submodules https://github.com/garys-gpu-benchmarks/bundle-nvidia-ubuntu-2604 /opt/benchmarks
cd /opt/benchmarks
./run_benchmark_suite.sh -w 418 --smoke     # just this workload
./run_benchmark_suite.sh --smoke            # all 32 workloads
./show_metrics.sh 418                       # this workload's runs from the ledger
```

The suite tools (`run_benchmark_suite.sh`, `get_benchmark_results.sh`, `list_runs.sh`, `show_metrics.sh`) are documented in [`gpu-bench-suite`](https://github.com/garys-gpu-benchmarks/gpu-bench-suite).

Other platforms: [AMD 24.04](https://github.com/garys-gpu-benchmarks/bundle-amd-ubuntu-2404) (101–132), [NVIDIA 24.04](https://github.com/garys-gpu-benchmarks/bundle-nvidia-ubuntu-2404) (201–232), [AMD 26.04](https://github.com/garys-gpu-benchmarks/bundle-amd-ubuntu-2604) (301–332).

## Troubleshooting

- **SSH dropped during setup:** reconnect and run `sudo bash setup.sh --assume-yes` again. It resumes; do not delete `.venv` or `.cache`.
- **A metric reads `na`:** the run could not measure it on this host. The reason is in the run's `metrics_summary.txt` and `errors_info.txt`; see also [Notes](#notes).

## Repository Layout

```text
.
├── setup.sh                      # installs dependencies and builds the benchmark
├── run_benchmark.sh              # runs one profile, parses, validates
├── show_metrics.sh               # prints this workload's ledger rows
├── list_runs.sh                  # prints every run in the ledger
├── SPEC.md                       # engineering specification
├── benchmark_specification.json  # workload definition
├── config/                       # profile values and thresholds
├── src/cudnn_conv.cu             # benchmark source
├── scripts/                      # build, collect, parse, validate
├── docs/                         # PRD and contracts
├── tests/
├── results/                      # run outputs
└── LICENSE
```

## Continuous Integration

- [CI](.github/workflows/ci.yml) runs lint and static checks on every pull request and push to `main`. It does not run the benchmark.
- [GPU Smoke Benchmark](.github/workflows/gpu-smoke.yml) runs a profile on a self-hosted runner labeled `gpu`, `nvidia`, `ubu2604`. It only starts by hand (**Actions → GPU Smoke Benchmark → Run workflow**), so pull requests and forks cannot run it.

Both workflows call shared steps in [`shared-workflows@v1`](https://github.com/garys-gpu-benchmarks/shared-workflows).
