trait Matrix {
  fn get_n_rows(&self)->usize;
  fn get_n_cols(&self)->usize;
  fn get(&self, row : usize, col : usize)->f32;
  fn set(&mut self, row : usize, col : usize, value : f32);
}

struct DenseMatrix {
  row_major_data : Vec<f32>, n_rows : usize, n_cols : usize
}

impl Matrix for DenseMatrix{
  fn get_n_rows(self
                : &DenseMatrix)
      ->usize{self.n_rows} fn
      get_n_cols(self
                 : &DenseMatrix)
      ->usize{self.n_cols} fn
      get(self
          : &DenseMatrix, row
          : usize, col
          : usize)
      ->f32 {
    assert !(row < self.get_n_rows(), col < self.get_n_cols());
    self.row_major_data[row * self.get_n_cols() + col]
  }

  fn set(self : &mut DenseMatrix, row : usize, col : usize, value : f32) {
    assert !(row < self.get_n_rows(), col < self.get_n_cols());
    let index = row * self.get_n_cols() + col;
    self.row_major_data[index] = value;
  }
}

fn create_all_zeros_dense_matrix(rows : usize, cols : usize)->DenseMatrix {
  DenseMatrix {
  row_major_data:
    vec ![0.0; rows * cols], n_rows : rows, n_cols : cols
  }
}

fn dense_matric_from_vector(data
                            : Vec<f32>, rows
                            : usize, cols
                            : usize)
    ->DenseMatrix {
  assert_eq !(rows * cols, data.len());
  DenseMatrix {
  row_major_data:
    vec ![0.0; rows * cols], n_rows : rows, n_cols : cols
  }
}

use std::collections::HashMap;

struct UnstructuredSparseMatrix {
  row_major_data : std::collections::HashMap<usize, f32>,
                   n_rows : usize,
                            n_cols : usize
}

fn create_all_zeros_ustructured_sparse_matrix
    (rows: usize, cols: usize) -> UnstructuredSparseMatrix{
  UnstructuredSparseMatrix {
  row_major_data:
    HashMap::new (), n_rows : rows, n_cols : cols
  }
}

impl Matrix for UnstructuredSparseMatrix{
  fn get_n_rows(self
                : &UnstructuredSparseMatrix)
      ->usize{self.n_rows} fn
      get_n_cols(self
                 : &UnstructuredSparseMatrix)
      ->usize{self.n_cols} fn
      get(self
          : &UnstructuredSparseMatrix, row
          : usize, col
          : usize)
      ->f32 {
    assert !(row < self.get_n_rows(), col < self.get_n_cols());
    let index = &(row * self.get_n_cols() + col);
    if
      self.row_major_data.contains_key(index) {
        return self.row_major_data[index];
      }
    return 0.0;
  }

  fn set(self
         : &mut UnstructuredSparseMatrix, row
         : usize, col
         : usize, value
         : f32) {
    assert !(row < self.get_n_rows(), col < self.get_n_cols());
    let index = row * self.get_n_cols() + col;
    self.row_major_data.insert(index, value);
  }
}

fn matmul(a : &Matrix, b : &Matrix)->DenseMatrix {
  assert !(a.get_n_cols() == b.get_n_rows());
  let result_n_rows = a.get_n_rows();
  let result_n_cols = b.get_n_cols();
  let mut result = DenseMatrix {
  row_major_data:
    vec ![0.0; result_n_rows * result_n_cols], n_rows : result_n_rows,
                                                        n_cols : result_n_cols
  };

    for
      result_row in 0..result_n_rows {
        for
          result_col in 0..result_n_cols {
            let mut value = 0.0;
            for
              summation_idx in 0..a.get_n_cols() {
                let a_element = a.get(result_row, summation_idx);
                let b_element = b.get(summation_idx, result_col);
                value += a_element * b_element;
              }
            result.set(result_row, result_col, value);
          }
      }
    result
}

#[cfg(test)]
mod tests {
  use super::*;

#[test]
  fn test_dense_matmul_0() {
    let a = DenseMatrix{
      row_major_data : vec ![ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 ],
      n_rows : 3,
      n_cols : 2
    };
    let b = DenseMatrix{
      row_major_data : vec ![ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 ],
      n_rows : 2,
      n_cols : 3
    };
    let result = matmul(&a, &b);
    assert_eq !(result.row_major_data,
                vec ![ 9.0, 12.0, 15.0, 19.0, 26.0, 33.0, 29.0, 40.0, 51.0 ]);
    assert_eq !(result.n_rows, 3);
    assert_eq !(result.n_cols, 3);
  }

#[test]
  fn test_dense_matmul_1() {
    let a = DenseMatrix{
      row_major_data : vec ![ 1., 2., 3., 4. ],
      n_cols : 2,
      n_rows : 2
    };
    let result = matmul(&a, &a);
    assert_eq !(result.row_major_data, vec ![ 7.0, 10.0, 15.0, 22.0 ]);
    assert_eq !(result.n_cols, 2);
    assert_eq !(result.n_rows, 2);
  }
#[test]
  fn test_dense_matmul_2() {
    let a = DenseMatrix{
      row_major_data : vec ![
        1.0, 2.0, 6.0, 2.0, 4.0, 6.0, 3.0, 8.0, 5.0, 8.0, 7.0, 2.0, 8.0, 6.0,
        9.0, 8.0, 3.0, 7.0, 0.0, 7.0, 1.0, 2.0, 1.0, 2.0, 8.0, 4.0, 2.0, 3.0
      ],
      n_rows : 4,
      n_cols : 7
    };
    let b = DenseMatrix{
      row_major_data : vec ![
        5.0, 4.0, 3.0, 6.0, 7.0, 3.0, 5.0, 0.0, 3.0, 3.0, 9.0, 8.0, 6.0, 4.0,
        6.0, 4.0, 6.0, 9.0, 6.0, 3.0, 7.0, 7.0, 9.0, 0.0, 9.0, 1.0, 8.0, 8.0
      ],
      n_rows : 7,
      n_cols : 4
    };
    let result = matmul(&a, &b);
    assert_eq !(result.row_major_data, vec ![
      142.0, 117.0, 181.0, 98.0, 263.0, 179.0, 295.0, 194.0, 210.0, 147.0,
      207.0, 114.0, 136.0, 102.0, 143.0, 96.0
    ]);
    assert_eq !(result.n_rows, 4);
    assert_eq !(result.n_cols, 4);
  }

#[test]
  fn test_dense_matmul_3() {
    let a = DenseMatrix{row_major_data : vec ![5.0], n_rows : 1, n_cols : 1};
    let b = DenseMatrix{row_major_data : vec ![9.0], n_rows : 1, n_cols : 1};
    let result = matmul(&a, &b);
    assert_eq !(result.row_major_data, vec ![45.0]);
    assert_eq !(result.n_rows, 1);
    assert_eq !(result.n_cols, 1);
  }

#[test]
  fn test_dense_matmul_4() {
    let a = DenseMatrix{row_major_data : vec ![5.0], n_rows : 1, n_cols : 1};
    let b = DenseMatrix{row_major_data : vec ![9.0], n_rows : 1, n_cols : 1};
    let result = matmul(&a, &b);
    assert_eq !(result.row_major_data, vec ![45.0]);
    assert_eq !(result.n_rows, 1);
    assert_eq !(result.n_cols, 1);
  }

#[test]
  fn test_dense_matmul_5() {
    let a = DenseMatrix{
      row_major_data : vec ![
        9.0, 9.0, 2.0, 0.0, 3.0, 3.0, 1.0, 0.0, 8.0, 6.0, 5.0, 7.0, 0.0, 6.0,
        3.0
      ],
      n_rows : 5,
      n_cols : 3
    };
    let b = DenseMatrix{
      row_major_data : vec ![
        5.0, 0.0, 8.0, 8.0, 3.0, 2.0, 1.0, 9.0, 4.0, 8.0, 2.0,
        2.0, 3.0, 4.0, 7.0, 2.0, 0.0, 8.0, 7.0, 3.0, 4.0
      ],
      n_rows : 3,
      n_cols : 7
    };
    let result = matmul(&a, &b);
    assert_eq !(result.row_major_data, vec ![
      140.0, 40.0, 144.0, 106.0, 59.0, 51.0, 53.0,  48.0, 18.0,
      24.0,  30.0, 27.0,  18.0,  24.0, 61.0, 16.0,  8.0,  72.0,
      59.0,  26.0, 33.0,  124.0, 34.0, 88.0, 114.0, 77.0, 48.0,
      54.0,  75.0, 30.0,  48.0,  36.0, 33.0, 27.0,  36.0
    ]);
    assert_eq !(result.n_rows, 5);
    assert_eq !(result.n_cols, 7);
  }

#[test]
  fn test_dense_matmul_6() {
    let a = DenseMatrix{
      row_major_data : vec ![ 4.0, 4.5, 1.0, 0.0, 2.0, 4.5, 3.5 ],
      n_rows : 7,
      n_cols : 1
    };
    let b = DenseMatrix{
      row_major_data : vec ![ 4.5, 1.5, 4.0, 1.5, 3.0, 1.5, 2.5, 4.0, 0.5 ],
      n_rows : 1,
      n_cols : 9
    };
    let result = matmul(&a, &b);
    assert_eq !(result.row_major_data, vec ![
      18.0, 6.0,   16.0, 6.0,  12.0,  6.0,  10.0, 16.0,  2.0,  20.25, 6.75,
      18.0, 6.75,  13.5, 6.75, 11.25, 18.0, 2.25, 4.5,   1.5,  4.0,   1.5,
      3.0,  1.5,   2.5,  4.0,  0.5,   0.0,  0.0,  0.0,   0.0,  0.0,   0.0,
      0.0,  0.0,   0.0,  9.0,  3.0,   8.0,  3.0,  6.0,   3.0,  5.0,   8.0,
      1.0,  20.25, 6.75, 18.0, 6.75,  13.5, 6.75, 11.25, 18.0, 2.25,  15.75,
      5.25, 14.0,  5.25, 10.5, 5.25,  8.75, 14.0, 1.75
    ]);
    assert_eq !(result.n_rows, 7);
    assert_eq !(result.n_cols, 9);
  }
}

fn main() {}
