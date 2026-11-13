# Results: 418 cuDNN Convolution Microbenchmark

**Main goal:** Measure cuDNN convolution solver speed  
**Tools:** Compile cudnn_conv.cu with nvcc -lcudnn; run bin/cudnn_conv twice via collect_cudnn_conv.py

Reference results measured by the maintainer on the system shown in each section (generator TEMPLATE_00_138 · recorded 2026-11-12). Your own runs write `results/RESULTS_ACTUAL.md` in the same format.

> Smoke runs use deliberately small sizes. They check that the workload installs and runs end to end; their numbers are not performance results. Baseline and extended are.

<!-- profile:smoke -->
## smoke

- **System:** 1× NVIDIA H100 80GB HBM3 (81559 MiB) · Intel(R) Xeon(R) Platinum 8470 · Ubuntu 26.04.1 LTS · kernel 6.8.0-147-generic · driver 595.91.07
- **Run:** 2026-11-12T23:09:56Z · status **PASS** (exit 0)
- **Time (mm:ss):** setup 00:50 · benchmark 00:03 · whole run 01:01
- **Source:** run 418_20261112_230956
- **Command:** `bash run_benchmark.sh --smoke --dtype FP16 --conv-direction all --batch-size 1 --input-channels 8 --input-height 16 --input-width 16 --output-channels 8 --kernel-height 3 --kernel-width 3 --stride-h 1 --stride-w 1 --group-count 1 --search-strategy find --warmup-iters 1 --num-iterations 2`

| # | Metric | Result |
|---|---|---|
| 1 | Convolution kernel execution time, forward (kernel_time_msec) | 0.023224 |
| 2 | Achieved convolution throughput, forward, TFLOP/s (tflops_fwd) | 0.012698500000000001 |
| 3 | Minimum forward operand traffic, GB/s (min_operand_traffic_gb_s) | 0.402344 |
| 4 | Solver search time (solver_search_msec), sum of cudnnFind wall time across the directions run | 215.3952525 |
<!-- /profile:smoke -->
