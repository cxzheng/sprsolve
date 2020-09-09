use cauchy::Scalar;
use super::{DenseVec, DenseVecMut};
use sprs::{CompressedStorage, CsMat, CsMatView};

/// An interface for the sparse matrix and dense vector multiplication.
pub trait MatVecMul<T: Scalar> {
    /// Multiply this matrix with the provided vector `v_in` and put the results
    /// in `v_out`.
    ///
    /// This method will check the dimension agreement and panick if the dimensions don't match.
    fn mul_vec<'a, IN: 'a + DenseVec<'a, T>, OUT: for<'b> DenseVecMut<'b, T>>(&self, v_in: IN, v_out: OUT);

    /// # Safety
    ///
    /// This method will not check the dimension agreement. If the dimensions don't match,
    /// they will result in *[undefined behavior](https://doc.rust-lang.org/reference/behavior-considered-undefined.html)*.
    unsafe fn mul_vec_unchecked<'a, IN: 'a + DenseVec<'a, T>, OUT: for<'b> DenseVecMut<'b, T>>(&self, v_in: IN, v_out: OUT);
}

// 'a refers to the lt of data in CSMatView
impl<'a, T: Scalar> MatVecMul<T> for CsMatView<'a, T> {
    #[inline]
    fn mul_vec<'va, IN: 'va + DenseVec<'va, T>, OUT: for<'vb> DenseVecMut<'vb, T>>(&self, v_in: IN, v_out: OUT) {
        if self.cols() != v_in.dim() || v_in.dim() != v_out.dim() {
            panic!("Dimension mismatch");
        }
        unsafe {
            self.mul_vec_unchecked(v_in, v_out);
        }
    }

    // This is very much identical to `mul_acc_mat_vec_csr` method provided in sprs crate.
    // Here 'va refers to the lt of data in DenseVec
    unsafe fn mul_vec_unchecked<'va, IN: 'va + DenseVec<'va, T>, OUT: for<'vb> DenseVecMut<'vb, T>>(&self, v_in: IN, mut v_out: OUT) {
        v_out.iter_mut().for_each(|v| *v = T::zero());

        match self.storage() {
            CompressedStorage::CSR => {
                for (row_ind, vec) in self.outer_iterator().enumerate() {
                    let t = v_out.get_unchecked_mut(row_ind);
                    for (col_ind, &value) in vec.iter() {
                        *t += *v_in.get_unchecked(col_ind) * value;
                    }
                }
            }
            CompressedStorage::CSC => {
                // initialize it
                for (col_ind, vec) in self.outer_iterator().enumerate() {
                    let multiplier = v_in.get_unchecked(col_ind);
                    for (row_ind, &value) in vec.iter() {
                        let t = v_out.get_unchecked_mut(row_ind);
                        *t += *multiplier * value;
                    }
                }
            }
        }
    }
}

impl<T: Scalar> MatVecMul<T> for CsMat<T> {
    #[inline]
    fn mul_vec<'va, IN: 'va + DenseVec<'va, T>, OUT: for<'vb> DenseVecMut<'vb, T>>(&self, v_in: IN, v_out: OUT) {
        self.view().mul_vec(v_in, v_out);
    }

    #[inline]
    unsafe fn mul_vec_unchecked<'va, IN: 'va + DenseVec<'va, T>, OUT: for<'vb> DenseVecMut<'vb, T>>(&self, v_in: IN, v_out: OUT) {
        self.view().mul_vec_unchecked(v_in, v_out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_csc_mat() {
        let indptr: &[usize] = &[0, 2, 4, 5, 6, 7];
        let indices: &[usize] = &[2, 3, 3, 4, 2, 1, 3];
        let data: &[f64] = &[
            0.35310881, 0.42380633, 0.28035896, 0.58082095, 0.53350123, 0.88132896, 0.72527863,
        ];

        let mat =
            CsMatView::new_view(CompressedStorage::CSC, (5, 5), indptr, indices, data).unwrap();
        let vector = vec![0.1, 0.2, -0.1, 0.3, 0.9];
        let mut res_vec = vec![0.; 5];
        mat.mul_vec(&vector, &mut res_vec);

        let expected_output = vec![0., 0.26439869, -0.01803924, 0.75120319, 0.11616419];

        let epsilon = 1e-8;

        assert!(res_vec
            .iter()
            .zip(expected_output.iter())
            .all(|(x, y)| (*x - *y).abs() < epsilon));
    }

    #[test]
    fn dense_csr_mat() {
        let indptr: &[usize] = &[0, 3, 3, 5, 6, 7];
        let indices: &[usize] = &[1, 2, 3, 2, 3, 4, 4];
        let data: &[f64] = &[
            0.75672424, 0.1649078, 0.30140296, 0.10358244, 0.6283315, 0.39244208, 0.57202407,
        ];

        let mat =
            CsMatView::new_view(CompressedStorage::CSR, (5, 5), indptr, indices, data).unwrap();
        let slice = vec![0.1, 0.2, -0.1, 0.3, 0.9];
        let mut res_vec = vec![0., 0., 0., 0., 0.];
        unsafe {
            mat.mul_vec_unchecked(&slice, &mut res_vec);
        }

        let expected_output = vec![0.22527496, 0., 0.17814121, 0.35319787, 0.51482166];

        let epsilon = 1e-8;

        assert!(res_vec
            .iter()
            .zip(expected_output.iter())
            .all(|(x, y)| (*x - *y).abs() < epsilon));
    }
}