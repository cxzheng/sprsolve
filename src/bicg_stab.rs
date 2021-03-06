//! An impl of BiCGSTAB solver.

use super::{error::*, vecalg::*, MatVecMul};
use cauchy::Scalar;
use num_traits::{float::*, Zero};
use std::{
    intrinsics::{likely, unlikely},
    ptr::copy_nonoverlapping,
    slice::from_raw_parts_mut,
};

/// Implementation of BiCGSTAB algorithm for solving non-symmetric positive indefinite
/// sparse linear system.
///
/// Note: consider to turn on `mkl` feature for improved performance.
#[allow(non_snake_case, non_camel_case_types)]
pub struct BiCGStab<'data, T: Scalar + Send + Sync, M: MatVecMul<T>> {
    A: &'data M,
    workspace: Vec<T>,
    size: usize,
}

impl<'data, T: Scalar + Send + Sync, M: MatVecMul<T>> BiCGStab<'data, T, M> {
    #[allow(non_snake_case)]
    pub fn new(A: &'data M, size: usize) -> Self {
        BiCGStab {
            A,
            workspace: vec![T::zero(); size * 7],
            size,
        }
    }

    /// Solves Ax = b, without preconditioner
    #[allow(clippy::many_single_char_names)]
    pub fn solve(
        &mut self,
        rhs: &[T],
        x: &mut [T],
        max_iter: usize,
        tol: T::Real,
    ) -> SolveResult<(usize, T::Real)> {
        let n = rhs.len();
        // check the format
        if n != self.size {
            return Err(SolverError::IncompatibleMatrixFormat(String::from(
                "Input vec dimension doesn't match the matrix size",
            )));
        }
        if n != x.len() {
            return Err(SolverError::IncompatibleMatrixFormat(String::from(
                "Input and output vec dimension do not match",
            )));
        }

        let rhs_norm = norm2(rhs);
        if unlikely(rhs_norm <= T::Real::epsilon()) {
            // when rhs = 0, x is set to zero.
            x.iter_mut().for_each(|v| *v = T::zero());
            return Ok((0, rhs_norm));
        }
        let tol2 = tol * rhs_norm;

        // Here is the internal memeory layout
        let ptr = self.workspace.as_mut_ptr();
        let r = unsafe { from_raw_parts_mut(ptr, n) }; // &mut [T]
        let r0 = unsafe { from_raw_parts_mut(ptr.add(n), n) };
        let y = unsafe { from_raw_parts_mut(ptr.add(2 * n), n) };
        let v = unsafe { from_raw_parts_mut(ptr.add(3 * n), n) };
        let t = unsafe { from_raw_parts_mut(ptr.add(4 * n), n) };
        // Without precond. we don't need s/z here.
        // let s_z = unsafe { from_raw_parts_mut(ptr.add(5 * n), n) }; // s / z
        unsafe {
            self.A.mul_vec_unchecked(x, &mut *r);
        }
        axpy(-T::one(), rhs, &mut *r); // r = A*x - rhs
        unsafe {
            // r0 = r
            copy_nonoverlapping(r.as_ptr(), r0.as_mut_ptr(), n);
        }
        let r0_norm = norm2(&*r0);
        if unlikely(r0_norm <= tol2) {
            return Ok((0, r0_norm / rhs_norm));
        }
        let mut r0_norm_tol = r0_norm * T::Real::epsilon();
        r0_norm_tol = r0_norm_tol * r0_norm_tol;

        // unroll the first iteration to initialize variables
        let mut rho = T::from_real(r0_norm * r0_norm); // rho != 0
        unsafe {
            // - y = r
            copy_nonoverlapping(r.as_ptr(), y.as_mut_ptr(), n);
            // - v = A*y
            self.A.mul_vec_unchecked(&*y, &mut *v);
        }
        // alpha = rho / r0.v
        let mut alpha = rho / conj_dot(&*r0, &*v);

        // - r = r - alpha * v
        // Now r is the s in the algorithm
        axpy(-alpha, &*v, &mut *r);

        // - t = A * z ==> t = A * r
        unsafe {
            self.A.mul_vec_unchecked(&*r, &mut *t);
        }
        // tmp = t.t
        let tmp = conj_dot(&*t, &*t);
        let mut w = if likely(tmp.re() > T::Real::zero()) {
            // w = t.s/tmp ==> w = t.r/tmp
            conj_dot(&*t, &*r) / tmp
        } else {
            T::zero()
        };
        // x = x - alpha*y - w*z
        axpy(-alpha, &*y, &mut *x);
        // Without precond: s_z (\hat s) is s, which is r
        axpy(-w, &*r, &mut *x);
        // r = s - w * t
        // now because r is the s, we have r = r - w*t
        axpy(-w, &*t, &mut *r);

        for its in 1..max_iter {
            let r_norm = norm2(&*r);
            if r_norm <= tol2 {
                return Ok((its, r_norm / rhs_norm));
            }
            let rho_old = rho;
            rho = conj_dot(&*r0, &*r);

            // Here r0_norm_tol has been squared
            if unlikely(rho.abs() < r0_norm_tol) {
                // r = A*x
                unsafe {
                    self.A.mul_vec_unchecked(x, &mut *r);
                }
                // r = A*x - rhs
                axpy(-T::one(), rhs, &mut *r);
                // r0 = r
                unsafe {
                    copy_nonoverlapping(r.as_ptr(), r0.as_mut_ptr(), n);
                }
                let r_norm = norm2(&*r);
                rho = T::from_real(r_norm * r_norm);
                r0_norm_tol = rho.re() * T::Real::epsilon() * T::Real::epsilon();
            }
            let beta = (rho / rho_old) * (alpha / w);

            /*
            axpy(-w, &*v, &mut *y); // y - w*v
            scale(beta, &mut *y);   // beta * (y-w*v)
            axpy(T::one(), &*r, &mut *y); // y = r + beta * (y - w*v)
            */
            // On multi-processor machine with iomp5 on, the following
            // code (using MKL's `axpby` extension) is faster.
            axpby(-beta * w, &*v, beta, &mut *y); // beta * (y - w*v)
            axpy(T::one(), &*r, &mut *y); // y = r + beta * (y - w*v)

            unsafe {
                // - v = A*y
                self.A.mul_vec_unchecked(&*y, &mut *v);
            }
            // alpha = rho / r0.v
            let tmp = conj_dot(&*r0, &*v);
            if unlikely(tmp.abs() <= T::Real::zero()) {
                //println!("{}", tmp);
                return Err(SolverError::BreakDown(its));
            }

            alpha = rho / tmp;
            // - r = r - alpha * v
            // Now r is the s in the algorithm
            axpy(-alpha, &*v, &mut *r);
            // - t = A * z ==> t = A * r
            unsafe {
                self.A.mul_vec_unchecked(&*r, &mut *t);
            }
            // tmp = t.t
            let tmp = conj_dot(&*t, &*t);
            w = if likely(tmp.re() > T::Real::zero()) {
                // NOTE: Here we could have used `mul_vec_dot` to combine t.r with A*r above.
                // But this won't be applicable for precond. BiCGStab method. So we don't do it for now.
                // w = t.s/tmp ==> w = t.r/tmp
                conj_dot(&*t, &*r) / tmp
            } else {
                T::zero()
            };
            // x = x - alpha*y - w*z
            axpy(-alpha, &*y, &mut *x); // x - alpha * y

            // without precond: s_z (\hat s) is s, which is r
            axpy(-w, &*r, &mut *x); // x - alpha*y - w*r

            //axpy(-w, &*s_z, &mut *x); // x - alpha*y - w*z
            // r = s - w * t
            // now because r is the s, we have r = r - w*t
            axpy(-w, &*t, &mut *r);
        }

        Err(SolverError::InsufficientIterNum(max_iter))
    }

    /// Solves Ax = b, with a preconditioner
    #[allow(clippy::many_single_char_names)]
    pub fn precond_solve<P: MatVecMul<T>>(
        &mut self,
        precond: &P,
        rhs: &[T],
        x: &mut [T],
        max_iter: usize,
        tol: T::Real,
    ) -> SolveResult<(usize, T::Real)> {
        let n = rhs.len();
        // check the format
        if n != self.size {
            return Err(SolverError::IncompatibleMatrixFormat(String::from(
                "Input vec dimension doesn't match the matrix size",
            )));
        }
        if n != x.len() {
            return Err(SolverError::IncompatibleMatrixFormat(String::from(
                "Input and output vec dimension do not match",
            )));
        }

        let rhs_norm = norm2(rhs);
        if unlikely(rhs_norm <= T::Real::epsilon()) {
            // when rhs = 0, x is set to zero.
            x.iter_mut().for_each(|v| *v = T::zero());
            return Ok((0, rhs_norm));
        }
        let tol2 = tol * rhs_norm;

        // Here is the internal memeory layout
        let ptr = self.workspace.as_mut_ptr();
        let r = unsafe { from_raw_parts_mut(ptr, n) }; // &mut [T]
        let r0 = unsafe { from_raw_parts_mut(ptr.add(n), n) };
        let y = unsafe { from_raw_parts_mut(ptr.add(2 * n), n) };
        let p = unsafe { from_raw_parts_mut(ptr.add(3 * n), n) };
        let v = unsafe { from_raw_parts_mut(ptr.add(4 * n), n) };
        let t = unsafe { from_raw_parts_mut(ptr.add(5 * n), n) };
        let z = unsafe { from_raw_parts_mut(ptr.add(6 * n), n) };

        unsafe {
            self.A.mul_vec_unchecked(x, &mut *r);
        }
        axpy(-T::one(), rhs, &mut *r); // r = A*x - rhs
        unsafe {
            // r0 = r
            copy_nonoverlapping(r.as_ptr(), r0.as_mut_ptr(), n);
        }
        let r0_norm = norm2(&*r0);
        if unlikely(r0_norm <= tol2) {
            return Ok((0, r0_norm / rhs_norm));
        }
        let mut r0_norm_tol = r0_norm * T::Real::epsilon();
        r0_norm_tol = r0_norm_tol * r0_norm_tol;

        // unroll the first iteration to initialize variables
        let mut rho = T::from_real(r0_norm * r0_norm); // rho != 0
        unsafe {
            copy_nonoverlapping(r.as_ptr(), p.as_mut_ptr(), n); // - p = r
            precond.mul_vec_unchecked(&*p, &mut *y); // - y = M^{-1}p
            self.A.mul_vec_unchecked(&*y, &mut *v); // - v = A*y
        }
        // alpha = rho / r0.v
        let mut alpha = rho / conj_dot(&*r0, &*v);
        // - r = r - alpha * v
        // Now r is the s in the algorithm
        axpy(-alpha, &*v, &mut *r);

        unsafe {
            // - z = M^{-1} r
            precond.mul_vec_unchecked(&*r, &mut *z);
            // - t = A * z
            self.A.mul_vec_unchecked(&*z, &mut *t);
        }
        // tmp = t.t
        let tmp = conj_dot(&*t, &*t);
        let mut w = if likely(tmp.re() > T::Real::zero()) {
            // w = t.s/tmp ==> w = t.r/tmp
            conj_dot(&*t, &*r) / tmp
        } else {
            T::zero()
        };

        // x = x - alpha*y - w*z
        // x = x - alpha*y
        axpy(-alpha, &*y, &mut *x);
        // s is r: x = x - w * z
        axpy(-w, &*z, &mut *x);
        // r = s - w * t
        // now because r is the s, we have r = r - w*t
        axpy(-w, &*t, &mut *r);

        for its in 1..max_iter {
            let r_norm = norm2(&*r);
            if r_norm <= tol2 {
                return Ok((its, r_norm / rhs_norm));
            }
            let rho_old = rho;
            rho = conj_dot(&*r0, &*r);

            // Here r0_norm_tol has been squared
            if unlikely(rho.abs() < r0_norm_tol) {
                // r = A*x
                unsafe {
                    self.A.mul_vec_unchecked(x, &mut *r);
                }
                // r = A*x - rhs
                axpy(-T::one(), rhs, &mut *r);
                // r0 = r
                unsafe {
                    copy_nonoverlapping(r.as_ptr(), r0.as_mut_ptr(), n);
                }
                let r_norm = norm2(&*r);
                rho = T::from_real(r_norm * r_norm);
                r0_norm_tol = rho.re() * T::Real::epsilon() * T::Real::epsilon();
            }
            let beta = (rho / rho_old) * (alpha / w);

            // p = r + beta * (p - w*v)
            // On multi-processor machine with iomp5 on, the following
            // code (using MKL's `axpby` extension) is faster.
            axpby(-beta * w, &*v, beta, &mut *p); // beta * (p - w*v)
            axpy(T::one(), &*r, &mut *p); // p = r + beta * (p - w*v)

            unsafe {
                precond.mul_vec_unchecked(&*p, &mut *y); // - y = M^{-1}p
                self.A.mul_vec_unchecked(&*y, &mut *v); // - v = A*y
            }
            // alpha = rho / r0.v
            let tmp = conj_dot(&*r0, &*v);
            if unlikely(tmp.abs() <= T::Real::zero()) {
                //println!("{}", tmp);
                return Err(SolverError::BreakDown(its));
            }

            alpha = rho / tmp;
            // - r = r - alpha * v
            // Now r is the s in the algorithm
            axpy(-alpha, &*v, &mut *r);
            unsafe {
                precond.mul_vec_unchecked(&*r, &mut *z); // - z = M^{-1} * r
                self.A.mul_vec_unchecked(&*z, &mut *t); // - t = A * z
            }
            // tmp = t.t
            let tmp = conj_dot(&*t, &*t);
            w = if likely(tmp.re() > T::Real::zero()) {
                conj_dot(&*t, &*r) / tmp
            } else {
                T::zero()
            };

            // x = x - alpha*y - w*z
            axpy(-alpha, &*y, &mut *x); // x - alpha * y
                                        // s is r: x = x - w * z
            axpy(-w, &*z, &mut *x);

            //axpy(-w, &*s_z, &mut *x); // x - alpha*y - w*z
            // r = s - w * t
            // now because r is the s, we have r = r - w*t
            axpy(-w, &*t, &mut *r);
        }

        Err(SolverError::InsufficientIterNum(max_iter))
    }
}
