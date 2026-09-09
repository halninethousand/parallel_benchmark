use image::{ImageBuffer, Luma};
use std::thread;
use std::time::Instant;

const MAX_ITERATIONS: u16 = 512;
const WIDTH: usize = 4000;
const HEIGHT: usize = 4000;
const THREAD_COUNT: usize = 4;

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
                re: -2.5 + (x as f64 / WIDTH as f64) * 3.5,
                im: -1.5 + (y as f64 / HEIGHT as f64) * 3.0,
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

    println!("=== Mandelbrot Benchmark ===");
    println!("Resolution:       {} x {}", WIDTH, HEIGHT);
    println!("Pixels:           {}", WIDTH * HEIGHT);
    println!("Max iterations:   {}", MAX_ITERATIONS);
    println!("Single-thread calculation: {:.3?}", single_thread_time);
    println!(
        "Parallel calculation ({} threads): {:.3?}",
        THREAD_COUNT, parallel_time
    );

    let single_thread_pixels_per_second =
        (WIDTH * HEIGHT) as f64 / single_thread_time.as_secs_f64();
    let parallel_pixels_per_second = (WIDTH * HEIGHT) as f64 / parallel_time.as_secs_f64();

    println!(
        "Single-thread rate: {:.2} million pixels/sec",
        single_thread_pixels_per_second / 1_000_000.0
    );
    println!(
        "Parallel rate:      {:.2} million pixels/sec",
        parallel_pixels_per_second / 1_000_000.0
    );
    println!(
        "Speedup:            {:.2}x",
        single_thread_time.as_secs_f64() / parallel_time.as_secs_f64()
    );

    let start_write_png = Instant::now();

    let mut image = ImageBuffer::<Luma<u16>, Vec<u16>>::new(WIDTH as u32, HEIGHT as u32);

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let value = parallel_pixels[y * WIDTH + x];
            // Rendering is separate from the benchmark: map the raw iteration
            // count to the full 16-bit grayscale range for a visible PNG.
            let scaled_value = u32::from(value) * u32::from(u16::MAX) / u32::from(MAX_ITERATIONS);
            let grayscale_value =
                u16::try_from(scaled_value).expect("scaled grayscale value must fit in u16");
            image.put_pixel(x as u32, y as u32, Luma([grayscale_value]));
        }
    }

    image.save("mandelbrot.png").unwrap();

    let write_time = start_write_png.elapsed();

    println!("PNG write time:   {:.3?}", write_time);
}
