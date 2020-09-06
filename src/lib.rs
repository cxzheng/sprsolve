/// An interface for the preconditioner.
/// 
/// A preconditioner $M$ can be viewed as a matrix $\mathbf{M}$ for which the solve for
/// $\mathbf{M}\mathbf{x} = \mathbf{b}$ is fast.
pub trait Precond {
}

/// An interface for the sparse matrix and dense vector multiplication.
pub trait MatVecMul {
}

pub fn gauss_seidel() {
}

pub fn conj_grad() {
}

pub fn precond_conj_grad() {
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
