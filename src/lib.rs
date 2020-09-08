//#![feature(min_const_generics)]

pub mod error;
mod gauss_seidel;
mod mat;

pub use gauss_seidel::*;
pub use mat::MatVecMul;

/// An interface for the preconditioner.
///
/// A preconditioner $M$ can be viewed as a matrix $\mathbf{M}$ for which the solve for
/// $\mathbf{M}\mathbf{x} = \mathbf{b}$ is fast.
pub trait Precond {}

pub fn conj_grad() {}

pub fn precond_conj_grad() {}
