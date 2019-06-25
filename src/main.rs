trait Matrix {
    fn get_n_rows(&self)->usize;
    fn get_n_cols(&self)->usize;
    fn get(&self, row:usize, col:usize) -> f32;
    fn set(&mut self, row:usize, col:usize, value: f32);
}

struct DenseMatrix{
    row_major_data: Vec<f32>,
    n_rows: usize,
    n_cols: usize
}

struct UnstructuredSparseMatrix{
    row_major_data: std::collections::HashMap<usize, f32>,
    n_rows: usize,
    n_cols: usize
}


impl Matrix for UnstructuredSparseMatrix{
    fn get_n_rows(self: &UnstructuredSparseMatrix)->usize{self.n_rows}
    fn get_n_cols(self: &UnstructuredSparseMatrix)->usize{self.n_cols}
    fn get(self: &UnstructuredSparseMatrix, row: usize, col: usize) -> f32{
        assert!(row < self.get_n_rows(), col < self.get_n_cols());
        self.row_major_data[&(row*self.get_n_cols() + col)]
    }

    fn set(self: &mut UnstructuredSparseMatrix, row: usize, col: usize, value: f32){
        assert!(row < self.get_n_rows(), col < self.get_n_cols());
        let index = row*self.get_n_cols() + col;
        self.row_major_data.insert(index, value);
    }
}

impl Matrix for DenseMatrix{
    fn get_n_rows(self: &DenseMatrix)->usize{self.n_rows}
    fn get_n_cols(self: &DenseMatrix)->usize{self.n_cols}
    fn get(self: &DenseMatrix, row: usize, col: usize) -> f32{
        assert!(row < self.get_n_rows(), col < self.get_n_cols());
        self.row_major_data[row*self.get_n_cols() + col]
    }

    fn set(self: &mut DenseMatrix, row: usize, col: usize, value: f32){
        assert!(row < self.get_n_rows(), col < self.get_n_cols());
        let index = row*self.get_n_cols() + col;
        self.row_major_data[index] = value;
    }
}


fn matmul(a: &Matrix, b: &Matrix) -> DenseMatrix {
    assert!(a.get_n_cols()==b.get_n_rows());
    let result_n_rows = a.get_n_rows();
    let result_n_cols = b.get_n_cols();
    let mut result = DenseMatrix{
        row_major_data: vec![0.0;result_n_rows * result_n_cols],
        n_rows: result_n_rows,
        n_cols: result_n_cols
    };

    for result_row in 0..result_n_rows{
        for result_col in 0..result_n_cols{
            let mut value = 0.0;
            for summation_idx in 0..b.get_n_cols(){
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
    fn test_matmul() {
        let a = DenseMatrix{
            row_major_data:vec![1., 2. ,3. ,4.],
            n_cols:2, 
            n_rows:2
        };
        let result = matmul(&a,&a);
        assert_eq!(result.row_major_data, vec![7.0, 10.0, 15.0, 22.0]);
    }
}

fn main() {
}
