use thiserror::Error;

pub type SolveResult<T> = std::result::Result<T, SolverError>;

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum SolverError {
    #[error("Incompatible input matrix format: {0}")]
    IncompatibleMatrixFormat(String),

    #[error("Matrix has zero diagonal element at {0}")]
    ZeorDiagonalElem(usize),

    #[error("Insufficient interation #: {0}")]
    InsufficientIterNum(usize),
}
