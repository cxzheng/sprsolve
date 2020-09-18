//! An impl of BiCGSTAB solver.

use super::{error::*, MatVecMul};
use cauchy::Scalar;
use num_traits::{float::*, Zero};
use sprs::CsMatView;

pub struct BiCGStab<'data, T: Scalar> {
    solver: BiCGStab_Backup<'data, T>,
}

/// The backup implementation of BiCGSTAB algorithm when no BLAS/MKL is
/// available, focusing on correctness not performance.
#[allow(non_snake_case, non_camel_case_types)]
struct BiCGStab_Backup<'data, T: Scalar> {
    A: CsMatView<'data, T>,
    workspace: Vec<T>,
}

impl<'data, T: Scalar> BiCGStab_Backup<'data, T> {
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
        Ok(BiCGStab_Backup {
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
        unimplemented!()
    }
}
