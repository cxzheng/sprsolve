use cauchy::Scalar;
use num::ToPrimitive;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use sprs::{CompressedStorage, CsMatI, CsMatViewI, SpIndex};
use std::{intrinsics::likely, slice::from_raw_parts};

/// An interface for the sparse matrix and dense vector multiplication.
///
/// # Performance Tuning
///
/// The _parallel_ feature turns on multi-thread computing in the [`mul_vec_unchecked`] using Rayon
pub trait MatVecMul<T: Scalar> {
    /// Multiply this matrix with the provided vector `v_in` and put the results
    /// in `v_out`.
    ///
    /// This method will check the dimension agreement and panick if the dimensions don't match.
    fn mul_vec(&self, v_in: &[T], v_out: &mut [T]);

    /// This is similar to `mkl_sparse_?_dotmv` method provided by MKL. It computes
    /// v_out = A*v_in
    /// and returns conj(v_in).dot(v_out)
    fn mul_vec_dot(&self, v_in: &[T], v_out: &mut [T]) -> T;

    /// # Safety
    ///
    /// This method will not check the dimension agreement. If the dimensions don't match,
    /// they will result in *[undefined behavior](https://doc.rust-lang.org/reference/behavior-considered-undefined.html)*.
    unsafe fn mul_vec_unchecked(&self, v_in: &[T], v_out: &mut [T]);

    /// The unchecked version of [`mul_vec_dot`]
    ///
    /// # Safety
    ///
    /// This method will not check the dimension agreement. If the dimensions don't match,
    /// they will result in *[undefined behavior](https://doc.rust-lang.org/reference/behavior-considered-undefined.html)*.
    unsafe fn mul_vec_dot_unchecked(&self, v_in: &[T], v_out: &mut [T]) -> T;
}

// 'a refers to the lt of data in CSMatView
impl<'a, T: Scalar + Send + Sync, I: SpIndex + ToPrimitive> MatVecMul<T> for CsMatViewI<'a, T, I> {
    #[inline]
    fn mul_vec(&self, v_in: &[T], v_out: &mut [T]) {
        if self.cols() != v_in.len() || v_in.len() != v_out.len() {
            panic!("Dimension mismatch");
        }
        unsafe {
            self.mul_vec_unchecked(v_in, v_out);
        }
    }

    #[inline]
    fn mul_vec_dot(&self, v_in: &[T], v_out: &mut [T]) -> T {
        if self.cols() != v_in.len() || v_in.len() != v_out.len() {
            panic!("Dimension mismatch");
        }
        unsafe { self.mul_vec_dot_unchecked(v_in, v_out) }
    }

    // This is very much identical to `mul_acc_mat_vec_csr` method provided in sprs crate.
    // Here 'vec refers to the lt of data in DenseVec
    unsafe fn mul_vec_unchecked(&self, v_in: &[T], v_out: &mut [T]) {
        // compiler will turn this into memset if needed
        debug_assert!(self.cols() == v_in.len() && v_in.len() == v_out.len());
        v_out.iter_mut().for_each(|v| *v = T::zero());

        // We don't use `match` here because the `likely` instrinsics leads to better
        // branch layout in the assembly code.
        if likely(self.storage() == CompressedStorage::CSR) {
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
                indptr
                    .par_windows(2)
                    .zip(v_out.par_iter_mut())
                    .for_each(|(row_range, row_ret)| {
                        let st = row_range.get_unchecked(0).to_usize().unwrap();
                        let nn = row_range.get_unchecked(1).to_usize().unwrap() - st;
                        let local_idx = from_raw_parts(index_ptr.0.add(st), nn); // directly construct slice to avoid bound check
                        let local_dat = from_raw_parts(data_ptr.0.add(st), nn);
                        *row_ret = local_idx.iter().zip(local_dat.iter()).fold(
                            T::zero(),
                            |acc, (&lid, &ldat)| {
                                acc + *v_in.get_unchecked(lid.to_usize().unwrap()) * ldat
                            },
                        );
                    });
            }
            #[cfg(not(feature = "parallel"))]
            {
                // most basic backup implementation
                let indptr = self.indptr();
                let index_ptr = self.indices().as_ptr();
                let data_ptr = self.data().as_ptr();
                indptr
                    .windows(2)
                    .zip(v_out.iter_mut())
                    .for_each(|(row_range, row_ret)| {
                        let st = row_range.get_unchecked(0).to_usize().unwrap();
                        let nn = row_range.get_unchecked(1).to_usize().unwrap() - st;
                        let local_idx = from_raw_parts(index_ptr.add(st), nn); // directly construct slice to avoid bound check
                        let local_dat = from_raw_parts(data_ptr.add(st), nn);
                        *row_ret = local_idx.iter().zip(local_dat.iter()).fold(
                            T::zero(),
                            |acc, (&lid, &ldat)| {
                                acc + *v_in.get_unchecked(lid.to_usize().unwrap()) * ldat
                            },
                        );
                    });
            }
        } else {
            // CSC
            // We dont' do any performance optimization for CSC yet.
            // As we haven't used it that much.
            // initialize it
            for (col_ind, vec) in self.outer_iterator().enumerate() {
                let multiplier = v_in.get_unchecked(col_ind);
                for (row_ind, &value) in vec.iter() {
                    let t = v_out.get_unchecked_mut(row_ind);
                    *t += *multiplier * value;
                }
            }
        }
    } // end fn

    unsafe fn mul_vec_dot_unchecked(&self, v_in: &[T], v_out: &mut [T]) -> T {
        use super::vecalg::conj_dot;

        self.mul_vec_unchecked(v_in, v_out);
        conj_dot(v_in, v_out)
    }
}

/// Wrap type to send the pointer across the thread
#[cfg(feature = "parallel")]
struct SendPtr<T: Send>(*const T);
#[cfg(feature = "parallel")]
unsafe impl<T: Send> Send for SendPtr<T> {}
#[cfg(feature = "parallel")]
unsafe impl<T: Send> Sync for SendPtr<T> {}

impl<T: Scalar + Send + Sync, I: SpIndex + 'static> MatVecMul<T> for CsMatI<T, I> {
    #[inline]
    fn mul_vec(&self, v_in: &[T], v_out: &mut [T]) {
        self.view().mul_vec(v_in, v_out);
    }

    #[inline]
    unsafe fn mul_vec_unchecked(&self, v_in: &[T], v_out: &mut [T]) {
        self.view().mul_vec_unchecked(v_in, v_out);
    }

    #[inline]
    fn mul_vec_dot(&self, v_in: &[T], v_out: &mut [T]) -> T {
        self.view().mul_vec_dot(v_in, v_out)
    }

    #[inline]
    unsafe fn mul_vec_dot_unchecked(&self, v_in: &[T], v_out: &mut [T]) -> T {
        self.view().mul_vec_dot_unchecked(v_in, v_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::CsMatView;
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

    #[test]
    fn dense_csr_mat_2() {
        let indptr: Vec<i32> = vec![0, 3, 3, 5, 6, 7];
        let indices: Vec<i32> = vec![1, 2, 3, 2, 3, 4, 4];
        let data = vec![
            0.75672424, 0.1649078, 0.30140296, 0.10358244, 0.6283315, 0.39244208, 0.57202407,
        ];

        let mat = CsMatI::new((5, 5), indptr, indices, data);
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
