//! A naive impl of Gauss-Seidel solver.
use super::{error::*, vecalg::*, MatVecMul};
use cauchy::Scalar;
use num_traits::{float::*, Zero};
use sprs::CsMatView;

#[allow(non_snake_case)]
pub struct GaussSeidel<'data, T: Scalar + PartialOrd + Send + Sync> {
    A: CsMatView<'data, T>,
    workspace: Vec<T>,
}

impl<'data, T: Scalar + PartialOrd + Send + Sync> GaussSeidel<'data, T> {
    #[allow(non_snake_case)]
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
        Ok(GaussSeidel {
            A,
            workspace: vec![T::zero(); A.rows() * 2],
        })
    }

    pub fn solve(
        &mut self,
        rhs: &[T],
        x: &mut [T],
        max_iter: usize,
        eps: T::Real,
    ) -> SolveResult<(usize, T::Real)> {
        // check the format
        if rhs.len() != self.A.rows() {
            return Err(SolverError::IncompatibleMatrixFormat(String::from(
                "Input vec dimension doesn't match the matrix size",
            )));
        }
        if rhs.len() != x.len() {
            return Err(SolverError::IncompatibleMatrixFormat(String::from(
                "Input and output vec dimension do not match",
            )));
        }

        if max_iter == 0 {
            return Err(SolverError::InsufficientIterNum(max_iter));
        }

        let n_rows = rhs.len();
        let mut b_norm: T::Real = Zero::zero();

        // unroll the first iteration to compute norm of b and cache the diagonals
        for (row_ind, vec) in self.A.outer_iterator().enumerate() {
            let mut sigma = T::zero();
            let mut diag = None;
            for (col_ind, val) in vec.iter() {
                if row_ind != col_ind {
                    unsafe {
                        sigma += (*val) * (*x.get_unchecked(col_ind));
                    }
                } else {
                    diag = Some(val);
                }
            }
            if diag == None {
                return Err(SolverError::ZeorDiagonalElem(row_ind));
            }
            let diag = diag.unwrap();
            if diag.square() < T::Real::epsilon() {
                return Err(SolverError::ZeorDiagonalElem(row_ind));
            }
            unsafe {
                // store the diag elem in the cache for later use
                *self.workspace.get_unchecked_mut(n_rows + row_ind) = *diag;
                let rhs_v = rhs.get_unchecked(row_ind);
                b_norm += rhs_v.square(); // accumulate 2-norm
                *x.get_unchecked_mut(row_ind) = (*rhs_v - sigma) / *diag;
            }
        }
        let tol2 = eps * num_traits::Float::sqrt(b_norm);

        unsafe {
            self.A.mul_vec_unchecked(x, &mut self.workspace[0..n_rows]);
        }
        // r = A*x - b
        //self.workspace
        //    .iter_mut()
        //    .zip(rhs.iter())
        //    .for_each(|(a, b)| *a -= *b);
        axpy(-T::one(), rhs, &mut self.workspace[..n_rows]);
        // |r|
        //let res = self
        //    .workspace
        //    .iter()
        //    .take(n_rows)
        //    .fold(T::Real::zero(), |acc, x| acc + x.square());
        let res = norm2(&self.workspace[..n_rows]);

        if res <= tol2 {
            return Ok((1, res));
        }

        for it in 1..max_iter {
            for (row_ind, vec) in self.A.outer_iterator().enumerate() {
                let mut sigma = T::zero();
                for (col_ind, val) in vec.iter() {
                    if row_ind != col_ind {
                        unsafe {
                            sigma += (*val) * (*x.get_unchecked(col_ind));
                        }
                    }
                }
                unsafe {
                    let diag = self.workspace.get_unchecked(n_rows + row_ind);
                    let rhs_v = rhs.get_unchecked(row_ind);
                    *x.get_unchecked_mut(row_ind) = (*rhs_v - sigma) / *diag;
                }
            }

            unsafe {
                self.A.mul_vec_unchecked(x, &mut self.workspace[0..n_rows]);
            }
            // r = A*x - b
            axpy(-T::one(), rhs, &mut self.workspace[..n_rows]);
            // |r|
            let res = norm2(&self.workspace[..n_rows]);

            if res <= tol2 {
                return Ok((it, res));
            }
        }
        Err(SolverError::InsufficientIterNum(max_iter))
    }
}
