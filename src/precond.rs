use super::MatVecMul;
use cauchy::Scalar;
use std::{marker::PhantomData, ops::Mul};

/// Diagonal preconditioner
pub struct DiagPrecond<T, V>
where
    T: Scalar + Mul<V, Output = T>,
    V: Scalar,
{
    diag_inv: Vec<V>,
    _marker: PhantomData<T>,
}

impl<T, V> DiagPrecond<T, V>
where
    T: Scalar + Mul<V, Output = T>,
    V: Scalar,
{
    pub fn new(diag: &[V]) -> Self {
        let mut diag_inv: Vec<V> = Vec::with_capacity(diag.len());
        for v in diag.iter() {
            diag_inv.push(V::one() / *v);
        }
        DiagPrecond {
            diag_inv,
            _marker: PhantomData,
        }
    }
}

impl<T, V> MatVecMul<T> for DiagPrecond<T, V>
where
    T: Scalar + Mul<V, Output = T>,
    V: Scalar,
{
    #[inline]
    fn mul_vec(&self, v_in: &[T], v_out: &mut [T]) {
        if self.diag_inv.len() != v_in.len() || self.diag_inv.len() != v_out.len() {
            panic!("Dimension mismatch");
        }
        unsafe {
            self.mul_vec_unchecked(v_in, v_out);
        }
    }

    #[inline]
    unsafe fn mul_vec_unchecked(&self, v_in: &[T], v_out: &mut [T]) {
        for (r, (v, s)) in v_out.iter_mut().zip(v_in.iter().zip(self.diag_inv.iter())) {
            *r = (*v) * (*s);
        }
    }

    #[inline]
    fn mul_vec_dot(&self, _v_in: &[T], _v_out: &mut [T]) -> T {
        unimplemented!()
    }

    #[inline]
    unsafe fn mul_vec_dot_unchecked(&self, _v_in: &[T], _v_out: &mut [T]) -> T {
        unimplemented!()
    }
}
