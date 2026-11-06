# PRD.md:  "The Why"; Product requirements, benchmark metadata table, high-level requirements, etc.

Product Requirements Document

"The Why"; Product requirements, benchmark metadata table, high-level requirements, etc. Defines the benchmark goal, validation objective, test name, benchmark number, category, and high-level success criteria.

## Benchmark Matrix Document Metadata (via benchmark_specification.json)

This PRD.md section is populated from benchmark_specification.json, which is the structured source of benchmark-specific product requirements.

## Workload Number
418

## Workload Name
cuDNN Convolution Microbenchmark

## Execution Summary (Run and Measure)
Compile src/cudnn_conv.cu with nvcc -lcudnn (requires cudnn.h on the default include path) and run bin/cudnn_conv twice on synthetic tensors. The harness runs FP16 tensor-op convolutions with same padding (kernel/2) and, per direction (fwd, bwd-data, bwd-weights), times cudnnFind algorithm search and the selected algorithm's kernel

## Main Goal
Measure cuDNN convolution solver speed

## Validation Objective
Validates cuDNN convolution timing and TFLOPS from bin/cudnn_conv. Requires a discoverable cudnn.h and libcudnn

## Workload Category
Compute & Math Kernels

## Validation Requirement

The benchmark must include an automated SQLite-integrated validation layer that verifies persisted results from `results/benchmark.db`. Validation must confirm:

1. The benchmark run completed successfully with no tool errors.
2. Required samples and aggregate metrics were persisted for every swept shape.
3. Metrics are finite and physically sensible (positive, within plausible bounds).
4. Measured values satisfy configured thresholds when the workload defines pass/fail gates.
5. The benchmark fails validation when required data is missing, invalid, or outside bounds.

## Non-Functional Requirements

| Requirement | Target |
|---|---|
| Automation | Runs to completion without manual intervention after `bash run_benchmark.sh` |
| Idempotency | Re-running `run_benchmark.sh` appends a new run; never corrupts existing rows |
| Persistence | All metrics survive script exit; `results/benchmark.db` is the durable record |
