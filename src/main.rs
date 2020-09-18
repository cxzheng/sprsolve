use sprsolve::vecalg::*;

fn main() {
    let a = vec![1_f64; 100];
    let b = vec![2_f64; 100];
    let r = dot(a.as_slice(), b.as_slice());
    println!("{}", r);

    let a = vec![2_f32; 100];
    let b = vec![3_f32; 100];
    let r = dot(a.as_slice(), b.as_slice());
    println!("{}", r);
}
