//! This module implements linear algebra operations on vectors.

use cauchy::Scalar;
use std::ops::{Deref, DerefMut, Mul};

/// compute $\mathbf{x}\cdot\mathbf{y}$.
/// 
/// *NOTE:* No conjugate is taken if the vector is complex-valued.
pub fn dot<T, IN1, IN2>(vec1: IN1, vec2: IN2) -> T
where
    T: Scalar,
    IN1: Deref<Target = [T]>,
    IN2: Deref<Target = [T]>,
{
    assert_eq!(vec1[..].len(), vec2[..].len());
    dot_fallback(&vec1[..], &vec2[..])
}

pub fn axpy<S, T, IN, OUT>(a: S, vec1: IN, mut vec2: OUT)
where
    S: Copy,
    T: Scalar + Mul<S, Output = T>,
    IN: Deref<Target = [T]>,
    OUT: DerefMut<Target = [T]>,
{
    assert_eq!(vec1[..].len(), vec2[..].len());
    axpy_fallback(a, &vec1[..], &mut vec2[..])
}

#[inline]
fn dot_fallback<T: Scalar>(vec1: &[T], vec2: &[T]) -> T {
    vec1.iter()
        .zip(vec2.iter())
        .fold(T::zero(), |acc, (x, y)| acc + (*x) * (*y))
}

#[inline]
fn axpy_fallback<S: Copy, T: Scalar + Mul<S, Output = T>>(a: S, vec1: &[T], vec2: &mut [T]) {
    vec2.iter_mut()
        .zip(vec1.iter())
        .for_each(|(y, x)| *y += *x * a);
}
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dot_generic() {
        let a: &[f64] = &[1., 1., 1., 1., 1., 1.];
        let b: &[f64] = &[1., 2., 3., 4., 5., 6.];
        approx::abs_diff_eq!(21., dot_fallback(a, b));

        let a = vec![1.; 6];
        let b = vec![1., 2., 3., 4., 5., 6.];
        approx::abs_diff_eq!(21., dot_fallback(&a, &b));
        println!("{:?}", b);
    }

    #[test]
    fn dot_generic_complex() {
        use cauchy::c64;
        let a = vec![c64::new(0., 1.); 6];
        let mut b: Vec<c64> = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            b.push(c64::new(i as f64, 0.));
        }
        let ret = dot_fallback(&a, &b);
        approx::abs_diff_eq!(0., ret.re);
        approx::abs_diff_eq!(21., ret.im);

        const N: usize = 8;
        let mut a: Vec<c64> = Vec::with_capacity(N);
        let mut b: Vec<c64> = Vec::with_capacity(N);
        let mut s = 0_f64;
        for i in 0..N {
            a.push(c64::new(i as f64, i as f64));
            b.push(c64::new(i as f64, -(i as f64)));
            s += (i * i) as f64;
        }
        let ret = dot_fallback(&a, &b);
        approx::abs_diff_eq!(s, ret.re);
        approx::abs_diff_eq!(0., ret.im);
    }

    #[test]
    fn axpy_generic_complex() {
        use cauchy::c64;
        let a = vec![c64::new(0., 1.); 6];
        let mut b: Vec<c64> = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            b.push(c64::new(i as f64, 0.));
        }
        axpy_fallback(1., &a, &mut b);
        for i in 0..a.len() {
            approx::abs_diff_eq!(i as f64, b[i].re);
            approx::abs_diff_eq!(1., b[i].im);
        }

        let mut b: Vec<c64> = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            b.push(c64::new(i as f64, 0.));
        }
        axpy_fallback(c64::new(0., 1.), &a, &mut b);
        for i in 0..a.len() {
            approx::abs_diff_eq!((i as f64) - 1., b[i].re);
            approx::abs_diff_eq!(0., b[i].im);
        }
    }

    #[test]
    fn axpy_generic_f32() {
        let a = vec![1_f32; 6];
        let mut b = vec![0_f32; 6];
        for _ in 0..4 {
            axpy_fallback(2_f32, &a, &mut b);
        }
        for i in 0..a.len() {
            approx::abs_diff_eq!(8., b[i]);
        }
    }
}
