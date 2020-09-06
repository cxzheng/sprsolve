#![feature(min_const_generics)]

mod mat;
mod vec;

pub use vec::{DenseVec, DenseVecMut};

/// An interface for the preconditioner.
///
/// A preconditioner $M$ can be viewed as a matrix $\mathbf{M}$ for which the solve for
/// $\mathbf{M}\mathbf{x} = \mathbf{b}$ is fast.
pub trait Precond {}

pub fn gauss_seidel() {}

pub fn conj_grad() {}

pub fn precond_conj_grad() {}
