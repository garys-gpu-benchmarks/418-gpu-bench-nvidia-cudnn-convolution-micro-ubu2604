# Results: 418 cuDNN Convolution Microbenchmark

**Main goal:** Measure cuDNN convolution solver speed  
**Tools:** Compile cudnn_conv.cu with nvcc -lcudnn; run bin/cudnn_conv twice via collect_cudnn_conv.py

Reference results measured by the maintainer on the system shown in each section (generator TEMPLATE_00_149 · recorded 2026-11-19). Your own runs write `results/RESULTS_ACTUAL.md` in the same format.

> Smoke runs use deliberately small sizes. They check that the workload installs and runs end to end; their numbers are not performance results. Baseline and extended are.

<!-- profile:smoke -->
## smoke

- **System:** 1× NVIDIA H100 NVL (95830 MiB) · Intel(R) Xeon(R) Platinum 8452Y · Ubuntu 26.04.1 LTS · kernel 6.8.0-138-generic · driver 595.112.07
- **Run:** 2026-11-19T21:51:58Z · status **PASS** (exit 0)
- **Time (mm:ss):** setup 01:01 · benchmark 00:03 · whole run 01:10
- **Source:** run 418_20261119_215158
- **Command:** `bash run_benchmark.sh --smoke --dtype FP16 --conv-direction all --batch-size 1 --input-channels 8 --input-height 16 --input-width 16 --output-channels 8 --kernel-height 3 --kernel-width 3 --stride-h 1 --stride-w 1 --group-count 1 --search-strategy find --warmup-iters 1 --num-iterations 2`

| # | Metric | Result |
|---|---|---|
| 1 | Convolution kernel execution time, forward (kernel_time_msec) | 0.03224 |
| 2 | Achieved convolution throughput, forward, TFLOP/s (tflops_fwd) | 0.009147 |
| 3 | Minimum forward operand traffic, GB/s (min_operand_traffic_gb_s) | 0.289826 |
| 4 | Solver search time (solver_search_msec), sum of cudnnFind wall time across the directions run | 292.263961 |
<!-- /profile:smoke -->
