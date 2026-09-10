use image::{ImageBuffer, Luma};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::thread;
use std::time::Instant;

const MAX_ITERATIONS: u16 = 512;
const WIDTH: usize = 5120;
const HEIGHT: usize = 2880;
// 5800X3D performance cutoff is at 32 os threads
const THREAD_COUNT: usize = 32;

const REAL_MIN: f64 = -2.5;
const REAL_SPAN: f64 = 3.5;
// Match the complex-plane aspect ratio to the 16:9 pixel grid so a unit of
// distance has the same size horizontally and vertically in the PNG
const IMAGINARY_SPAN: f64 = REAL_SPAN * HEIGHT as f64 / WIDTH as f64;
const IMAGINARY_MIN: f64 = -IMAGINARY_SPAN / 2.0;

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

fn main() {
    // Allocate before either timer so allocation is not part of the benchmark.
    let mut single_thread_pixels = vec![0u16; WIDTH * HEIGHT];
    let mut parallel_pixels = vec![0u16; WIDTH * HEIGHT];

    let single_thread_start = Instant::now();
    calculate_rows(&mut single_thread_pixels, 0, HEIGHT);
    let single_thread_time = single_thread_start.elapsed();

    let parallel_start = Instant::now();
    calculate_parallel(&mut parallel_pixels, THREAD_COUNT);
    let parallel_time = parallel_start.elapsed();

    // This happens after both timers and confirms the two implementations agree.
    assert_eq!(single_thread_pixels, parallel_pixels);

    // Create the reusable Rayon pool outside the calculation timer.
    let rayon_pool = ThreadPoolBuilder::new()
        .num_threads(THREAD_COUNT)
        .build()
        .unwrap();

    let rayon_start = Instant::now();
    calculate_rayon(&mut parallel_pixels, &rayon_pool);
    let rayon_time = rayon_start.elapsed();

    // Reusing the buffer is safe because the previous calculation has ended.
    assert_eq!(single_thread_pixels, parallel_pixels);

    println!("=== Mandelbrot Benchmark ===");
    println!("Resolution:       {} x {}", WIDTH, HEIGHT);
    println!("Pixels:           {}", WIDTH * HEIGHT);
    println!("Max iterations:   {}", MAX_ITERATIONS);

    let single_thread_pixels_per_second =
        (WIDTH * HEIGHT) as f64 / single_thread_time.as_secs_f64();
    let parallel_pixels_per_second = (WIDTH * HEIGHT) as f64 / parallel_time.as_secs_f64();
    let rayon_pixels_per_second = (WIDTH * HEIGHT) as f64 / rayon_time.as_secs_f64();

    println!("\n--- Single-threaded baseline ---");
    println!("Calculation time:  {:.3?}", single_thread_time);
    println!(
        "Pixels/sec:        {:.2} million",
        single_thread_pixels_per_second / 1_000_000.0
    );

    println!("\n--- std::thread ({} workers) ---", THREAD_COUNT);
    println!("Calculation time:  {:.3?}", parallel_time);
    println!(
        "Pixels/sec:        {:.2} million",
        parallel_pixels_per_second / 1_000_000.0
    );
    println!(
        "Speedup:           {:.2}x",
        single_thread_time.as_secs_f64() / parallel_time.as_secs_f64()
    );

    println!("\n--- Rayon ({} workers) ---", THREAD_COUNT);
    println!("Calculation time:  {:.3?}", rayon_time);
    println!(
        "Pixels/sec:        {:.2} million",
        rayon_pixels_per_second / 1_000_000.0
    );
    println!(
        "Speedup:           {:.2}x",
        single_thread_time.as_secs_f64() / rayon_time.as_secs_f64()
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
