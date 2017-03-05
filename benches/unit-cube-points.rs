//! 10 thousand uniformly distributed points in the unit cube.
#![feature(test)]
extern crate beebox;
extern crate beevage;
extern crate cgmath;
extern crate rand;
extern crate test;
use test::black_box;

use beebox::Aabb;
use cgmath::{Vector3, vec3};
use rand::{ChaChaRng, Rng, SeedableRng};

// Fractional part of Pi in 16 bit
const SEED: [u32; 8] = [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344, 0xa4093822, 0x299f31d0,
                        0x082efa98, 0xec4e6c89];
struct Point(Vector3<f32>);

impl beevage::Primitive for Point {
    fn bounding_box(&self) -> beebox::Aabb {
        Aabb::from_corners(self.0, self.0)
    }
}

fn create_points(n: usize) -> Vec<Point> {
    let mut rng = ChaChaRng::from_seed(&SEED);
    let mut points = Vec::with_capacity(n);
    for _ in 0..n {
        points.push(Point(vec3(rng.next_f32(), rng.next_f32(), rng.next_f32())));
    }
    points
}

fn build(points: &[Point]) {
    let config = beevage::Config {
        bucket_count: 16,
        traversal_cost: 1.0,
        max_depth: 64,
    };
    let bb = Aabb::new(points.iter().map(|p| p.0));
    black_box(beevage::binned_sah(config, &points, bb));
}

#[bench]
fn bench_1k(b: &mut test::Bencher) {
    let points = create_points(1_000);
    b.iter(|| build(&points))
}

#[bench]
fn bench_10k(b: &mut test::Bencher) {
    let points = create_points(10_000);
    b.iter(|| build(&points))
}

#[bench]
fn bench_100k(b: &mut test::Bencher) {
    let points = create_points(100_000);
    b.iter(|| build(&points))
}
