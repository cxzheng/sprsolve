use thiserror::Error;

pub type SolveResult<T> = std::result::Result<T, SolverError>;

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum SolverError {
    #[error("Incompatible input matrix format: {0}")]
    IncompatibleMatrixFormat(String),
}
