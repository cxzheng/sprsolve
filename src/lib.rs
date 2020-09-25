//#![feature(min_const_generics)]
#![feature(core_intrinsics)]

mod bicg_stab;
pub mod error;
mod gauss_seidel;
mod mat;
#[cfg(feature = "mkl")]
mod mkl_mat;
pub mod vecalg;

pub use bicg_stab::BiCGStab;
pub use gauss_seidel::*;
pub use mat::MatVecMul;
#[cfg(feature = "mkl")]
pub use mkl_mat::*;

#[cfg(feature = "mkl")]
use std::any::TypeId;

/// Return `true` if `A` and `B` are the same type
#[cfg(feature = "mkl")]
#[inline(always)]
fn same_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

// Read pointer to type `A` as type `B`.
//
// **Panics** if `A` and `B` are not the same type
#[cfg(feature = "mkl")]
#[inline(always)]
fn cast_as<A: 'static + Copy, B: 'static + Copy>(a: &A) -> B {
    debug_assert!(same_type::<A, B>());
    unsafe { std::ptr::read(a as *const _ as *const B) }
}
