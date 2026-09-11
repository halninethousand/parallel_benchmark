use image::{ImageBuffer, Luma};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::thread;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;

const MAX_ITERATIONS: u16 = 512;
const WIDTH: usize = 5120;
const HEIGHT: usize = 2880;
const DEFAULT_THREADS: usize = 32;
const DEFAULT_RUNS: usize = 1;
const TOKIO_TASKS_PER_WORKER: usize = 4;

const REAL_MIN: f64 = -2.5;
const REAL_SPAN: f64 = 3.5;
// Match the complex-plane aspect ratio to the 16:9 pixel grid so a unit of
// distance has the same size horizontally and vertically in the PNG
const IMAGINARY_SPAN: f64 = REAL_SPAN * HEIGHT as f64 / WIDTH as f64;
const IMAGINARY_MIN: f64 = -IMAGINARY_SPAN / 2.0;

struct Config {
    threads: usize,
    runs: usize,
}

struct Statistics {
    minimum: Duration,
    median: Duration,
    mean: Duration,
}

fn parse_positive_usize(value: String, argument: &str) -> usize {
    let parsed = value
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("{argument} must be a positive whole number"));

    assert!(parsed > 0, "{argument} must be greater than zero");
    parsed
}

fn parse_config() -> Config {
    let mut config = Config {
        threads: DEFAULT_THREADS,
        runs: DEFAULT_RUNS,
    };
    let mut arguments = std::env::args().skip(1);

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--threads" => {
                let value = arguments
                    .next()
                    .unwrap_or_else(|| panic!("--threads requires a value"));
                config.threads = parse_positive_usize(value, "--threads");
            }
            "--runs" => {
                let value = arguments
                    .next()
                    .unwrap_or_else(|| panic!("--runs requires a value"));
                config.runs = parse_positive_usize(value, "--runs");
            }
            "--help" | "-h" => {
                println!("Usage: cargo run --release -- [--threads N] [--runs N]");
                std::process::exit(0);
            }
            _ => panic!("unknown argument: {argument}"),
        }
    }

    config
}

fn calculate_statistics(times: &[Duration]) -> Statistics {
    let mut sorted_times = times.to_vec();
    sorted_times.sort_unstable();

    let middle = sorted_times.len() / 2;
    let median = if sorted_times.len() % 2 == 0 {
        Duration::from_secs_f64(
            (sorted_times[middle - 1].as_secs_f64() + sorted_times[middle].as_secs_f64()) / 2.0,
        )
    } else {
        sorted_times[middle]
    };
    let mean = Duration::from_secs_f64(
        times.iter().map(Duration::as_secs_f64).sum::<f64>() / times.len() as f64,
    );

    Statistics {
        minimum: sorted_times[0],
        median,
        mean,
    }
}

fn print_statistics(title: &str, statistics: &Statistics, baseline_median: Option<Duration>) {
    let pixels_per_second = (WIDTH * HEIGHT) as f64 / statistics.median.as_secs_f64();

    println!("\n--- {title} ---");
    println!("Minimum time:      {:.3?}", statistics.minimum);
    println!("Median time:       {:.3?}", statistics.median);
    println!("Mean time:         {:.3?}", statistics.mean);
    println!(
        "Median pixels/sec: {:.2} million",
        pixels_per_second / 1_000_000.0
    );

    if let Some(baseline) = baseline_median {
        println!(
            "Median speedup:    {:.2}x",
            baseline.as_secs_f64() / statistics.median.as_secs_f64()
        );
    }
}

#[derive(Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}

fn square_complex(z: Complex) -> Complex {
    Complex {
        re: z.re * z.re - z.im * z.im,
        im: 2.0 * z.re * z.im,
    }
}

// `pixels` contains rows start_y..end_y, rather than necessarily the whole image.
// That lets each thread receive exclusive access to just its own rows.
fn calculate_rows(pixels: &mut [u16], start_y: usize, end_y: usize) {
    for y in start_y..end_y {
        for x in 0..WIDTH {
            let c = Complex {
                re: REAL_MIN + (x as f64 / WIDTH as f64) * REAL_SPAN,
                im: IMAGINARY_MIN + (y as f64 / HEIGHT as f64) * IMAGINARY_SPAN,
            };

            let mut z = Complex { re: 0.0, im: 0.0 };

            let mut iterations = 0;

            while iterations < MAX_ITERATIONS {
                z = square_complex(z);

                z.re += c.re;
                z.im += c.im;

                if z.re * z.re + z.im * z.im > 4.0 {
                    break;
                }

                iterations += 1;
            }

            let local_y = y - start_y;
            pixels[local_y * WIDTH + x] = iterations;
        }
    }
}

fn calculate_parallel(pixels: &mut [u16], thread_count: usize) {
    // More workers than rows would only create empty jobs
    let worker_count = thread_count.clamp(1, HEIGHT);
    let rows_per_worker = HEIGHT.div_ceil(worker_count);
    let pixels_per_worker = rows_per_worker * WIDTH;

    // Scoped threads may borrow `pixels`. Ordinary `thread::spawn` requires
    // `'static` data, which would not allow these borrowed slices.
    thread::scope(|scope| {
        for (worker_index, worker_pixels) in pixels.chunks_mut(pixels_per_worker).enumerate() {
            let start_y = worker_index * rows_per_worker;
            let end_y = (start_y + rows_per_worker).min(HEIGHT);

            scope.spawn(move || {
                calculate_rows(worker_pixels, start_y, end_y);
            });
        }
    });
}

fn calculate_rayon(pixels: &mut [u16], pool: &ThreadPool) {
    pool.install(|| {
        // Each item is one row. Rayon can split this work further and let an
        // idle worker steal remaining rows from a busy worker.
        pixels
            .par_chunks_mut(WIDTH)
            .enumerate()
            .for_each(|(y, row_pixels)| calculate_rows(row_pixels, y, y + 1));
    });
}

fn tokio_task_count(worker_count: usize) -> usize {
    worker_count
        .saturating_mul(TOKIO_TASKS_PER_WORKER)
        .clamp(1, HEIGHT)
}

fn calculate_tokio(pixels: &mut [u16], runtime: &Runtime, worker_count: usize) {
    let task_count = tokio_task_count(worker_count);

    runtime.block_on(async {
        let mut handles = Vec::with_capacity(task_count);

        for task_index in 0..task_count {
            let start_y = task_index * HEIGHT / task_count;
            let end_y = (task_index + 1) * HEIGHT / task_count;
            let mut chunk_pixels = vec![0u16; (end_y - start_y) * WIDTH];

            // spawn_blocking requires an owned, 'static closure. Each task
            // therefore owns its chunk and returns it after calculation.
            handles.push(tokio::task::spawn_blocking(move || {
                calculate_rows(&mut chunk_pixels, start_y, end_y);
                (start_y, chunk_pixels)
            }));
        }

        for handle in handles {
            let (start_y, chunk_pixels) = handle.await.unwrap();
            let start_pixel = start_y * WIDTH;
            pixels[start_pixel..start_pixel + chunk_pixels.len()].copy_from_slice(&chunk_pixels);
        }
    });
}

fn main() {
    let config = parse_config();

    // Allocate before either timer so allocation is not part of the benchmark.
    let mut single_thread_pixels = vec![0u16; WIDTH * HEIGHT];
    let mut parallel_pixels = vec![0u16; WIDTH * HEIGHT];
    let rayon_pool = ThreadPoolBuilder::new()
        .num_threads(config.threads)
        .build()
        .unwrap();
    let tokio_runtime = tokio::runtime::Builder::new_current_thread()
        .max_blocking_threads(config.threads)
        .build()
        .unwrap();

    let mut single_thread_times = Vec::with_capacity(config.runs);
    let mut std_thread_times = Vec::with_capacity(config.runs);
    let mut rayon_times = Vec::with_capacity(config.runs);
    let mut tokio_times = Vec::with_capacity(config.runs);

    for run in 0..config.runs {
        let single_thread_start = Instant::now();
        calculate_rows(&mut single_thread_pixels, 0, HEIGHT);
        single_thread_times.push(single_thread_start.elapsed());

        // Rotate the parallel order to reduce a consistent warm-up advantage.
        for offset in 0..3 {
            match (run + offset) % 3 {
                0 => {
                    let std_thread_start = Instant::now();
                    calculate_parallel(&mut parallel_pixels, config.threads);
                    std_thread_times.push(std_thread_start.elapsed());
                }
                1 => {
                    let rayon_start = Instant::now();
                    calculate_rayon(&mut parallel_pixels, &rayon_pool);
                    rayon_times.push(rayon_start.elapsed());
                }
                2 => {
                    let tokio_start = Instant::now();
                    calculate_tokio(&mut parallel_pixels, &tokio_runtime, config.threads);
                    tokio_times.push(tokio_start.elapsed());
                }
                _ => unreachable!(),
            }

            assert_eq!(single_thread_pixels, parallel_pixels);
        }
    }

    let single_thread_statistics = calculate_statistics(&single_thread_times);
    let std_thread_statistics = calculate_statistics(&std_thread_times);
    let rayon_statistics = calculate_statistics(&rayon_times);
    let tokio_statistics = calculate_statistics(&tokio_times);

    println!("=== Mandelbrot Benchmark ===");
    println!("Resolution:       {} x {}", WIDTH, HEIGHT);
    println!("Pixels:           {}", WIDTH * HEIGHT);
    println!("Max iterations:   {}", MAX_ITERATIONS);
    println!("Worker threads:   {}", config.threads);
    println!("Runs per mode:    {}", config.runs);

    print_statistics("Single-threaded baseline", &single_thread_statistics, None);
    print_statistics(
        &format!("std::thread ({} workers)", config.threads),
        &std_thread_statistics,
        Some(single_thread_statistics.median),
    );
    print_statistics(
        &format!("Rayon ({} workers)", config.threads),
        &rayon_statistics,
        Some(single_thread_statistics.median),
    );
    print_statistics(
        &format!(
            "Tokio spawn_blocking ({} worker limit, {} jobs)",
            config.threads,
            tokio_task_count(config.threads)
        ),
        &tokio_statistics,
        Some(single_thread_statistics.median),
    );

    let mut image = ImageBuffer::<Luma<u16>, Vec<u16>>::new(WIDTH as u32, HEIGHT as u32);

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let value = parallel_pixels[y * WIDTH + x];
            // 16-bit grayscale value.
            let scaled_value = u32::from(value) * u32::from(u16::MAX) / u32::from(MAX_ITERATIONS);
            let grayscale_value =
                u16::try_from(scaled_value).expect("scaled grayscale value must fit in u16");
            image.put_pixel(x as u32, y as u32, Luma([grayscale_value]));
        }
    }

    image.save("mandelbrot.png").unwrap();
}
