use {ordered_float::OrderedFloat, std::ops};

pub type fx = f64;

#[derive(Clone)]
pub struct Mat {
  pub size: usize,
  pub data: Vec<fx>,
}

impl Mat {
  pub fn from_vec(size: usize, data: Vec<fx>) -> Self {
    assert!(!data.is_empty());
    assert_eq!(data.len(), size * size);
    Mat { size, data }
  }

  fn swap_rows(&mut self, i: usize, j: usize) {
    if i != j {
      for k in 0..self.size {
        self.data.swap(i * self.size + k, j * self.size + k);
      }
    }
  }

  pub fn mul_vec(&self, x: &[fx]) -> Vec<fx> {
    let n = self.size;
    (0..n).map(|i| (0..n).map(|j| self[(i, j)] * x[j]).sum()).collect()
  }
}

impl ops::Index<(usize, usize)> for Mat {
  type Output = fx;
  fn index(&self, (row, col): (usize, usize)) -> &fx {
    &self.data[row * self.size + col]
  }
}

impl ops::IndexMut<(usize, usize)> for Mat {
  fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut fx {
    &mut self.data[row * self.size + col]
  }
}

fn ord(x: fx) -> OrderedFloat<fx> {
  OrderedFloat(x)
}

#[unsafe(no_mangle)]
pub fn gauss(
  mut a @ Mat { size: n, .. }: Mat,
  mut b: Vec<fx>,
) -> Option<Vec<fx>> {
  assert_eq!(n, b.len());

  let mut scale: Vec<_> = a
    .data
    .chunks(n)
    .map(|row| unsafe {
      row.iter().copied().map(fx::abs).max_by(fx::total_cmp).unwrap_unchecked()
    })
    .collect();

  for k in 0..n {
    let pivot = unsafe {
      (k..n).max_by_key(|&i| ord(a[(i, k)].abs() / scale[i])).unwrap_unchecked()
    };

    if k != pivot {
      scale.swap(k, pivot);
      a.swap_rows(k, pivot);
      b.swap(k, pivot);
    }

    for i in k + 1..n {
      let f = a[(i, k)] / a[(k, k)];
      for j in k + 1..n {
        a[(i, j)] -= a[(k, j)] * f;
      }
      b[i] -= b[k] * f;
    }
  }

  for i in (0..n).rev() {
    for j in i + 1..n {
      b[i] -= a[(i, j)] * b[j];
    }
    b[i] /= a[(i, i)];
  }
  Some(b)
}

pub fn jacobi(a: &Mat, b: &[fx], eps: fx, iters: usize) -> Option<Vec<fx>> {
  let n = a.size;
  let mut x = vec![0.0; n];
  for _ in 0..iters {
    let new: Vec<_> = (0..n)
      .map(|i| {
        (b[i]
          - (0..n).filter(|&j| j != i).map(|j| a[(i, j)] * x[j]).sum::<fx>())
          / a[(i, i)]
      })
      .collect();
    if x.iter().zip(&new).map(|(xi, xni)| (xi - xni).abs()).sum::<fx>() < eps {
      return Some(new);
    }
    x = new;
  }
  None
}
