use num_complex::{Complex32, Complex64};
pub trait Scalar : cauchy::Scalar {
}

impl Scalar for f32 {

}
impl Scalar for f64 {

}
impl Scalar for Complex32 {

}
impl Scalar for Complex64 {

}