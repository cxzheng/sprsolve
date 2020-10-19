use num_complex::Complex64;

#[test]
#[ignore]
fn test_minres_complex() {
    let (rows, cols) = (8, 8);
    let (lap, rhs) = grid_laplacian((rows, cols));

    println!(
        "grid laplacian nnz structure:\n{}",
        sprs::visu::nnz_pattern_formatter(lap.view()),
    );
    assert!(sprs::is_symmetric(&lap));

    let mut x = vec![Complex64::default(); rows * cols];
    let mut solver = sprsolve::MinRes::new(&lap, lap.cols());
    //let mut solver = sprsolve::BiCGStab::new(&lap, lap.cols());
    let (iters, res) = solver
        .solve(rhs.as_slice(), x.as_mut_slice(), 300, 1E-22)
        .unwrap();
    println!(
        "Solved system in {} iterations with residual error {}",
        iters,
        res.sqrt()
    );
    for i in 0..rows {
        for j in 0..cols {
            print!("{} ", x[i * rows + j]);
        }
        println!("");
    }
}

#[inline]
fn val<T: num_traits::ToPrimitive>(row: T, col: T) -> Complex64 {
    Complex64::new(row.to_f64().unwrap(), col.to_f64().unwrap())
}

fn grid_laplacian(shape: (usize, usize)) -> (sprs::CsMat<Complex64>, Vec<Complex64>) {
    let (rows, cols) = shape;
    let n = rows * cols;
    let mut rhs = vec![Complex64::new(0., 0.); n];
    let mut ret_a = sprs::TriMat::<Complex64>::new((n, n));

    for i in 0..rows {
        for j in 0..cols {
            let vid = i * cols + j;
            let mut rv = Complex64::new(0., 0.);

            let c = Complex64::new(-4., -4.);
            ret_a.add_triplet(vid, vid, c);
            rv += c * val(i, j);

            let c = Complex64::new(1., 0.5);
            if i > 0 {
                let tid = (i - 1) * cols + j;
                ret_a.add_triplet(vid, tid, c);
                rv += c * val(i-1, j);
            } 

            if j > 0 {
                let tid = i * cols + j - 1;
                ret_a.add_triplet(vid, tid, c);
                rv += c * val(i, j - 1);
            } 

            if i < rows - 1 {
                let tid = (i + 1) * cols + j;
                ret_a.add_triplet(vid, tid, c);
                rv += c * val(i+1, j);
            } 

            if j < cols - 1 {
                let tid = i * cols + j + 1;
                ret_a.add_triplet(vid, tid, c);
                rv += c * val(i, j+1);
            } 

            rhs[vid] = rv;
        } // end for
    } // end for
    (ret_a.to_csr(), rhs)
}
