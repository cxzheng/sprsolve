pub trait DenseVec<T> {
    /// The dimension of the vector
    fn dim(&self) -> usize;

    fn val(&self, id: usize) -> Option<&T>;

    unsafe fn val_unchecked(&self, id: usize) -> &T;
}

pub trait DenseVecMut<T>: DenseVec<T> {
    fn val_mut(&mut self, id: usize) -> Option<&mut T>;

    unsafe fn val_uncheck_mut(&mut self, id: usize) -> &mut T;
}

// ---------------------------------------------------------------------
macro_rules! dense_vec_impl {
    ( < $( $gen:tt ),+ >, $elem:tt, $vec_type:ty) => {
        impl<$( $gen ),+> DenseVec<$elem> for $vec_type {
            #[inline]
            fn dim(&self) -> usize {
                self.len()
            }

            #[inline]
            fn val(&self, id: usize) -> Option<&$elem> {
                self.get(id)
            }

            #[inline]
            unsafe fn val_unchecked(&self, id: usize) -> &$elem {
                self.get_unchecked(id)
            }
        }
    }
}
macro_rules! dense_vec_mut_impl {
    ( < $( $gen:tt ),+ >, $elem:tt, $vec_type:ty) => {
        impl<$( $gen ),+> DenseVecMut<$elem> for $vec_type {

            #[inline]
            fn val_mut(&mut self, id: usize) -> Option<&mut $elem> {
                self.get_mut(id)
            }

            #[inline]
            unsafe fn val_uncheck_mut(&mut self, id: usize) -> &mut $elem {
                self.get_unchecked_mut(id)
            }
        }
    }
}

dense_vec_impl! {<T>, T, [T]}
dense_vec_impl! {<T>, T, Vec<T>}
dense_vec_impl! {<'a, T>, T, &'a [T]}
dense_vec_impl! {<'a, T>, T, &'a mut[T]}

dense_vec_mut_impl! {<T>, T, [T]}
dense_vec_mut_impl! {<'a, T>, T, &'a mut[T]}
dense_vec_mut_impl! {<T>, T, Vec<T>}

impl<T, const N: usize> DenseVec<T> for [T; N] {
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

impl<T, const N: usize> DenseVecMut<T> for [T; N] {

    #[inline]
    fn val_mut(&mut self, id: usize) -> Option<&mut T> {
        self.get_mut(id)
    }

    #[inline]
    unsafe fn val_uncheck_mut(&mut self, id: usize) -> &mut T {
        self.get_unchecked_mut(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_vec_1() {
        let a = vec![3, 4, 1, 2];
        assert_eq!(a.val(1).unwrap(), &4);
        assert_eq!(a[..].val(0).unwrap(), &3);

        let b = [3, 4, 1, 2];
        assert_eq!(b.val(1).unwrap(), &4);
        assert_eq!(b[..].val(0).unwrap(), &3);
    }
}
