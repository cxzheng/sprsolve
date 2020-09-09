pub trait DenseVec<'a, T: 'a> { // Here we need T to have a lt of 'a to allow the associated type
    type Iter: Iterator<Item = &'a T>;

    fn iter(&'a self) -> Self::Iter;

    /// The dimension of the vector
    fn dim(&self) -> usize;

    fn get(&self, id: usize) -> Option<&T>;

    unsafe fn get_unchecked(&self, id: usize) -> &T;
}

pub trait DenseVecMut<'a, T: 'a>: DenseVec<'a, T> {
    type IterMut: Iterator<Item = &'a mut T>;

    fn iter_mut(&'a mut self) -> Self::IterMut;

    fn get_mut(&mut self, id: usize) -> Option<&mut T>;

    unsafe fn get_unchecked_mut(&mut self, id: usize) -> &mut T;
}

// ---------------------------------------------------------------------
macro_rules! dense_vec_impl {
    ( $elem:tt, $vec_type:ty, $iter_type:ty ) => {
        impl<'a, $elem: 'a> DenseVec<'a, $elem> for $vec_type {
            type Iter = $iter_type;

            #[inline]
            fn iter(&'a self) -> Self::Iter {
                <[$elem]>::iter(self)
            }

            #[inline]
            fn dim(&self) -> usize {
                self.len()
            }

            #[inline]
            fn get(&self, id: usize) -> Option<&$elem> {
                <[$elem]>::get(self, id)
            }

            #[inline]
            unsafe fn get_unchecked(&self, id: usize) -> &$elem {
                <[$elem]>::get_unchecked(self, id)
            }
        }
    }
}

macro_rules! dense_vec_mut_impl {
    ( $elem:tt, $vec_type:ty, $iter_type:ty ) => {
        impl<'a, $elem: 'a> DenseVecMut<'a, $elem> for $vec_type {
            type IterMut = $iter_type;

            #[inline]
            fn iter_mut(&'a mut self) -> Self::IterMut {
                <[$elem]>::iter_mut(self)
            }

            #[inline]
            fn get_mut(&mut self, id: usize) -> Option<&mut $elem> {
                <[$elem]>::get_mut(self, id)
            }

            #[inline]
            unsafe fn get_unchecked_mut(&mut self, id: usize) -> &mut $elem {
                <[$elem]>::get_unchecked_mut(self, id)
            }
        }
    }
}

// Note: we purposely exclude the stack-allocated array so that the
// DenseVector can be moved with little cost.
//dense_vec_impl! {<T>, T, [T]}
//dense_vec_mut_impl! {<T>, T, [T]}

dense_vec_impl! {T, &[T], std::slice::Iter<'a, T>} 
dense_vec_impl! {T, Vec<T>, std::slice::Iter<'a, T>} 
dense_vec_impl! {T, &Vec<T>, std::slice::Iter<'a, T>} 
dense_vec_impl! {T, &mut Vec<T>, std::slice::Iter<'a, T>} 
dense_vec_impl! {T, &mut [T], std::slice::Iter<'a, T>}

dense_vec_mut_impl! {T, &mut[T], std::slice::IterMut<'a, T>}
dense_vec_mut_impl! {T, Vec<T>, std::slice::IterMut<'a, T>}
dense_vec_mut_impl! {T, &mut Vec<T>, std::slice::IterMut<'a, T>}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_vec_1() {
        let a = vec![3, 4, 1, 2];
        assert_eq!(a.get(1).unwrap(), &4);
        assert_eq!(a.as_slice().get(0).unwrap(), &3);
        assert_eq!(a.get(5), None);
        unsafe {
            assert_eq!(a.get_unchecked(1), &4);
        }

        let mut a = vec![3, 4, 1, 2];
        a.iter_mut().for_each(|v| *v += 1);
        a.iter_mut().for_each(|v| *v += 2);
        assert_eq!(a.get(0).unwrap(), &6);
        assert_eq!(a.get(1).unwrap(), &7);
        assert_eq!(a.get(2).unwrap(), &4);
        /*
        let b = [3, 4, 1, 2];
        assert_eq!(b.val(1).unwrap(), &4);
        assert_eq!(b[..].val(0).unwrap(), &3);
        */
    }
}

/* the following impl. requires #![feature(min_const_generics)]
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

*/
