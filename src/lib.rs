use std::iter::IntoIterator;

/// An interface for the preconditioner.
/// 
/// A preconditioner $M$ can be viewed as a matrix $\mathbf{M}$ for which the solve for
/// $\mathbf{M}\mathbf{x} = \mathbf{b}$ is fast.
pub trait Precond {
}

pub trait DenseVec<T> {
    /// The dimension of the vector
    fn dim(&self) -> usize;

    fn val(&self, id: usize) -> Option<&T>;

    unsafe fn val_unchecked(&self, id: usize) -> &T;
}

pub trait DenseVecMut<T> : DenseVec<T> {
    fn val_mut(&mut self, id: usize) -> Option<&mut T>;

    unsafe fn val_uncheck_mut(&mut self, id: usize) -> &mut T;
}

// ---------------------------------------------------------------------
impl <T> DenseVec<T> for [T] {
    #[inline]
    fn dim(&self) -> usize {
        self.len()
    }

    #[inline]
    fn val(&self, id: usize) -> Option<&T> {
        self.get(id)
    }

    #[inline]
    unsafe fn val_unchecked(&self, id: usize) -> &T {
        self.get_unchecked(id)
    }
}

impl <T> DenseVec<T> for Vec<T> {
    #[inline]
    fn dim(&self) -> usize {
        self.len()
    }

    #[inline]
    fn val(&self, id: usize) -> Option<&T> {
        self.get(id)
    }

    #[inline]
    unsafe fn val_unchecked(&self, id: usize) -> &T {
        self.get_unchecked(id)
    }
}

impl<'a, T: 'a> DenseVec<T> for &'a [T] {

    #[inline]
    fn dim(&self) -> usize {
        self.len()
    }

    #[inline]
    fn val(&self, id: usize) -> Option<&T> {
        self.get(id)
    }

    #[inline]
    unsafe fn val_unchecked(&self, id: usize) -> &T {
        self.get_unchecked(id)
    }
}

impl<'a, T: 'a> DenseVec<T> for &'a mut [T] {

    #[inline]
    fn dim(&self) -> usize {
        self.len()
    }

    #[inline]
    fn val(&self, id: usize) -> Option<&T> {
        self.get(id)
    }

    #[inline]
    unsafe fn val_unchecked(&self, id: usize) -> &T {
        self.get_unchecked(id)
    }
}

impl<'a, T: 'a> DenseVecMut<T> for &'a mut [T] {

    #[inline]
    fn val_mut(&mut self, id: usize) -> Option<&mut T> {
        self.get_mut(id)
    }

    #[inline]
    unsafe fn val_uncheck_mut(&mut self, id: usize) -> &mut T {
        self.get_unchecked_mut(id)
    }
}

/// An interface for the sparse matrix and dense vector multiplication.
pub trait MatVecMul<T> {
    fn mul_vec<IN: DenseVec<T>, OUT: DenseVecMut<T>>(v_in: &IN, v_out: &mut OUT);

    fn mul_vec_unchecked<IN: DenseVec<T>, OUT: DenseVecMut<T>>(v_in: &IN, v_out: &OUT);
}

pub fn gauss_seidel() {
}

pub fn conj_grad() {
}

pub fn precond_conj_grad() {
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_vec_1() {
        let a = vec![3, 4, 1, 2];
        assert_eq!(a.val(1).unwrap(), &4);
        assert_eq!(a[..].val(0).unwrap(), &3);
    }
}
