#![feature(let_chains)]

use {
  matc::{Mat, fx, gauss, jacobi},
  std::iter,
};

mod random {
  use {crate::*, rand::*};

  pub fn mat(size: usize) -> Mat {
    Mat::from_vec(
      size,
      (0..size * size).map(|_| random_range(-10.0..10.0)).collect(),
    )
  }
}

fn diagonally_dominant(mat: &Mat) -> bool {
  let mut diagonal = 0.0;
  for i in 0..mat.size {
    diagonal += mat[(i, i)].abs();
  }
  mat.data.iter().copied().map(fx::abs).sum::<fx>() - diagonal < diagonal
}

fn random(size: usize) {
  let a = loop {
    if let mat = random::mat(size)
    // && diagonally_dominant(&mat)
    {
      break mat;
    }
  };

  let x_true: Vec<fx> =
    (0..size).map(|_| rand::random_range(-10.0..10.0)).collect();

  let b = a.mul_vec(&x_true);

  let time = std::time::Instant::now();

  // let solution = jacobi(&a, &b, 0.0001, u32::MAX as usize);
  // println!("{solution:?}");

  // let solution1 = gauss(a.clone(), b.clone());
  let solution2 = gauss(a, b);

  // assert_eq!(solution1, solution2);
  // println!("{solution:?}");

  println!("time: {:?}", time.elapsed());
}

fn diff(a: &[fx], b: &[fx], eps: fx) {
  for (a, b) in iter::zip(a, b) {
    assert!(fx::abs(a - b) < eps, "{a} ~ {b}");
  }
}

fn test(size: usize) {
  let eps = 0.0001;

  let x: Vec<fx> = (0..size).map(|_| rand::random_range(-10.0..10.0)).collect();
  let a = loop {
    if let mat = random::mat(size)
    // && diagonally_dominant(&mat)
    {
      break mat;
    }
  };
  let b = a.mul_vec(&x);

  let js = jacobi(&a, &b, eps, 1024);
  let Some(s) = gauss(a, b.clone()) else { unreachable!() };

  diff(&s, &x, eps);
  if let Some(js) = js {
    diff(&js, &x, eps);
  }
}

use indicatif::ProgressIterator;

fn main() {
  for _ in (0..1024).progress() {
    test(100);
  }
}
