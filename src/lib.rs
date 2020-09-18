//#![feature(min_const_generics)]

pub mod error;
mod gauss_seidel;
mod mat;
pub mod vecalg;

pub use gauss_seidel::*;
pub use mat::MatVecMul;

use std::any::TypeId;

/// Return `true` if `A` and `B` are the same type
#[inline(always)]
fn same_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

// Read pointer to type `A` as type `B`.
//
// **Panics** if `A` and `B` are not the same type
#[inline(always)]
fn cast_as<A: 'static + Copy, B: 'static + Copy>(a: &A) -> B {
    debug_assert!(same_type::<A, B>());
    unsafe { ::std::ptr::read(a as *const _ as *const B) }
}

/// An interface for the preconditioner.
///
/// A preconditioner $M$ can be viewed as a matrix $\mathbf{M}$ for which the solve for
/// $\mathbf{M}\mathbf{x} = \mathbf{b}$ is fast.
pub trait Precond {}

pub fn conj_grad() {}

pub fn precond_conj_grad() {}
