use image::{GrayImage, Luma};
use std::time::Instant;

const MAX_ITERATIONS: u8 = 255;
const WIDTH: usize = 4000;
const HEIGHT: usize = 4000;

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

fn calculate_rows(pixels: &mut [u8], start_y: usize, end_y: usize) {
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

            pixels[y * WIDTH + x] = iterations;
        }
    }
}

fn main() {
    let mut pixels = vec![0u8; WIDTH * HEIGHT];

    let start = Instant::now();

    calculate_rows(&mut pixels, 0, HEIGHT);

    let calculation_time = start.elapsed();

    println!("=== Mandelbrot Benchmark ===");
    println!("Resolution:       {} x {}", WIDTH, HEIGHT);
    println!("Pixels:           {}", WIDTH * HEIGHT);
    println!("Max iterations:   {}", MAX_ITERATIONS);
    println!("Threads:          1");
    println!("Calculation time: {:.3?}", calculation_time);

    let pixels_per_second = (WIDTH * HEIGHT) as f64 / calculation_time.as_secs_f64();

    println!(
        "Pixels/sec:       {:.2} million",
        pixels_per_second / 1_000_000.0
    );

    let start_write_png = Instant::now();

    let mut image = GrayImage::new(WIDTH as u32, HEIGHT as u32);

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let value = pixels[y * WIDTH + x];
            image.put_pixel(x as u32, y as u32, Luma([value]));
        }
    }

    image.save("mandelbrot.png").unwrap();

    let write_time = start_write_png.elapsed();

    println!("PNG write time:   {:.3?}", write_time);
}
