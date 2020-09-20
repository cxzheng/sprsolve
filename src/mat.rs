use cauchy::Scalar;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use sprs::{CompressedStorage, CsMat, CsMatView};
#[cfg(feature = "parallel")]
use std::slice::from_raw_parts;

/// An interface for the sparse matrix and dense vector multiplication.
pub trait MatVecMul<T: Scalar> {
    /// Multiply this matrix with the provided vector `v_in` and put the results
    /// in `v_out`.
    ///
    /// This method will check the dimension agreement and panick if the dimensions don't match.
    fn mul_vec(&self, v_in: &[T], v_out: &mut [T]);

    /// # Safety
    ///
    /// This method will not check the dimension agreement. If the dimensions don't match,
    /// they will result in *[undefined behavior](https://doc.rust-lang.org/reference/behavior-considered-undefined.html)*.
    unsafe fn mul_vec_unchecked(&self, v_in: &[T], v_out: &mut [T]);
}

// 'a refers to the lt of data in CSMatView
impl<'a, T: Scalar + Send + Sync> MatVecMul<T> for CsMatView<'a, T> {
    #[inline]
    fn mul_vec(&self, v_in: &[T], v_out: &mut [T]) {
        if self.cols() != v_in.len() || v_in.len() != v_out.len() {
            panic!("Dimension mismatch");
        }
        unsafe {
            self.mul_vec_unchecked(v_in, v_out);
        }
    }

    // This is very much identical to `mul_acc_mat_vec_csr` method provided in sprs crate.
    // Here 'vec refers to the lt of data in DenseVec
    unsafe fn mul_vec_unchecked(&self, v_in: &[T], v_out: &mut [T]) {
        // compiler will turn this into memset if needed
        debug_assert!(self.cols() == v_in.len() && v_in.len() == v_out.len());
        v_out.iter_mut().for_each(|v| *v = T::zero());

        match self.storage() {
            CompressedStorage::CSR => {
                /*
                if cfg!(target_feature = "avx") {
                    if super::same_type::<T, f64>() {
                        // AVX + f64 implmentaion
                    }
                }
                */
                // back up implementation
                // When `parallel` is enabled, use rayon to parallelize the outer index iteration
                #[cfg(feature = "parallel")]
                {
                    let indptr = self.indptr();
                    let index_ptr = SendPtr(self.indices().as_ptr());
                    let data_ptr = SendPtr(self.data().as_ptr());
                    indptr.par_windows(2).zip(v_out.par_iter_mut()).for_each(
                        |(row_range, row_ret)| {
                            let st = *row_range.get_unchecked(0);
                            let nn = *row_range.get_unchecked(1) - st;
                            let local_idx = from_raw_parts(index_ptr.0.add(st), nn); // directly construct slice to avoid bound check
                            let local_dat = from_raw_parts(data_ptr.0.add(st), nn);
                            for (lid, ldat) in local_idx.iter().zip(local_dat.iter()) {
                                *row_ret += *v_in.get_unchecked(*lid) * (*ldat);
                            }
                        },
                    );
                }
                #[cfg(not(feature = "parallel"))]
                {
                    // most basic backup implementation
                    for (vec, ret) in self.outer_iterator().zip(v_out.iter_mut()) {
                        for (col_ind, &value) in vec.iter() {
                            *ret += *v_in.get_unchecked(col_ind) * value;
                        }
                    }
                }
            }
            // We dont' do any performance optimization for CSC yet.
            // As we haven't used it that much.
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
        } // end match
    } // end fn
}

/// Wrap type to send the pointer across the thread
struct SendPtr<T: Send>(*const T);
unsafe impl<T: Send> Send for SendPtr<T> {}
unsafe impl<T: Send> Sync for SendPtr<T> {}

impl<T: Scalar + Send + Sync> MatVecMul<T> for CsMat<T> {
    #[inline]
    fn mul_vec(&self, v_in: &[T], v_out: &mut [T]) {
        self.view().mul_vec(v_in, v_out);
    }

    #[inline]
    unsafe fn mul_vec_unchecked(&self, v_in: &[T], v_out: &mut [T]) {
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
