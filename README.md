# Mandelbrot Parallel Benchmark

A CPU-bound Mandelbrot renderer for comparing Rust parallelism approaches:

- Single-threaded baseline
- `std::thread`
- Rayon
- Tokio `spawn_blocking`

## Run

Use release mode for meaningful timings:

```powershell
cargo run --release -- --threads 32 --runs 10
```

Arguments:

- `--threads N`: CPU-worker limit for the parallel implementations. Default: `32`.
- `--runs N`: Timed measurements per implementation. Default: `1`.

The generated image is written to `mandelbrot.png`.

## Example: Ryzen 7 5800X3D

```text
=== Mandelbrot Benchmark ===
Resolution:       5120 x 2880
Pixels:           14745600
Max iterations:   512
Worker threads:   32
Runs per mode:    10

--- Single-threaded baseline ---
Minimum time:      3.873s
Median time:       3.883s
Mean time:         3.887s
Median pixels/sec: 3.80 million

--- std::thread (32 workers) ---
Minimum time:      321.519ms
Median time:       362.219ms
Mean time:         373.144ms
Median pixels/sec: 40.71 million
Median speedup:    10.72x

--- Rayon (32 workers) ---
Minimum time:      277.266ms
Median time:       278.372ms
Mean time:         278.548ms
Median pixels/sec: 52.97 million
Median speedup:    13.95x

--- Tokio spawn_blocking (32 worker limit, 128 jobs) ---
Minimum time:      282.763ms
Median time:       288.410ms
Mean time:         287.949ms
Median pixels/sec: 51.13 million
Median speedup:    13.46x
```
