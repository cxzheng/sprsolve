use num_traits::float::*;
use super::error::*;
use super::{DenseVec, DenseVecMut, Scalar};
use sprs::CsMatView;

/// A naive impl of Gauss-Seidel solver.
pub struct GaussSeidel<'data, T: Scalar> {
    A: CsMatView<'data, T>,
}

impl<'data, T: Scalar> GaussSeidel<'data, T> {
    pub fn new(A: CsMatView<'data, T>) -> SolveResult<Self> {
        if A.rows() != A.cols() {
            return Err(SolverError::IncompatibleMatrixFormat(String::from(
                "Not a square matrix",
            )));
        }

        if !A.is_csr() {
            return Err(SolverError::IncompatibleMatrixFormat(String::from(
                "Not in CSR format",
            )));
        }
        Ok(GaussSeidel { A })
    }

    pub fn solve<'a, IN: DenseVec<'a, T>, OUT: DenseVecMut<'a, T>>(
        &self,
        rhs: IN,
        x: &mut OUT,
        max_iter: usize,
        eps: T::Real,
    ) -> SolveResult<T::Real> {
        // check the format
        if rhs.dim() != self.A.rows() {
            return Err(SolverError::IncompatibleMatrixFormat(String::from(
                "Input vec dimension doesn't match the matrix size",
            )));
        }
        if rhs.dim() != x.dim() {
            return Err(SolverError::IncompatibleMatrixFormat(String::from(
                "Input and output vec dimension do not match",
            )));
        }

        /*
        for (row_ind, vec) in self.A.outer_iterator().enumerate() {
            let mut sigma = 0.;
            let mut diag = None;
            for (col_ind, &val) in vec.iter() {
                if row_ind != col_ind {
                    sigma += val * x[[col_ind]];
                } else {
                    diag = Some(val);
                }
            }
            // Gauss-Seidel requires a non-zero diagonal, which
            // is satisfied for a laplacian matrix
            let diag = diag.unwrap();
            let cur_rhs = rhs[[row_ind]];
            x[[row_ind]] = (cur_rhs - sigma) / diag;
        }
        */

        Ok(T::Real::neg_zero())
    }
}
