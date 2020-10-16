#[test]
fn test_minres() {
    let (rows, cols) = (8, 8);
    let (lap, rhs) = grid_laplacian((rows, cols));

    println!(
        "grid laplacian nnz structure:\n{}",
        sprs::visu::nnz_pattern_formatter(lap.view()),
    );
    assert!(sprs::is_symmetric(&lap));

    let mut x = vec![0_f64; rows * cols];
    let mut solver = sprsolve::MinRes::new(&lap, lap.cols());
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

#[test]
fn minres_ident() {
    let (rows, cols) = (8, 8);
    let (lap, rhs) = simple((rows, cols));

    println!(
        "grid laplacian nnz structure:\n{}",
        sprs::visu::nnz_pattern_formatter(lap.view()),
    );
    assert!(sprs::is_symmetric(&lap));

    let mut x = vec![0_f64; rows * cols];
    let mut solver = sprsolve::MinRes::new(&lap, lap.cols());
    let (iters, res) = solver
        .solve(rhs.as_slice(), x.as_mut_slice(), 300, 1E-20)
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

fn simple(shape: (usize, usize)) -> (sprs::CsMat<f64>, Vec<f64>) {
    let (rows, cols) = shape;
    let n = rows * cols;
    let mut ret_a = sprs::TriMat::<f64>::new((n, n));
    let mut rhs = vec![0_f64; n];
    for (i, v) in rhs.iter_mut().enumerate() {
        *v = (i + 1) as f64;
    }
    for i in 0..n {
        ret_a.add_triplet(i, i, ((i + 1) * 2) as f64);
    }
    (ret_a.to_csr(), rhs)
}

fn grid_laplacian(shape: (usize, usize)) -> (sprs::CsMat<f64>, Vec<f64>) {
    let (rows, cols) = shape;
    let n = rows * cols;
    let mut rhs = vec![0_f64; n];
    let mut ret_a = sprs::TriMat::<f64>::new((n, n));

    let bv = |row: isize, col: isize| (row + col) as f64;

    for i in 0..rows {
        for j in 0..cols {
            let vid = i * cols + j;

            ret_a.add_triplet(vid, vid, -4.);

            if i > 0 {
                let tid = (i - 1) * cols + j;
                ret_a.add_triplet(vid, tid, 1.);
            } else {
                rhs[vid] -= bv(i as isize - 1, j as isize);
            }

            if j > 0 {
                let tid = i * cols + j - 1;
                ret_a.add_triplet(vid, tid, 1.);
            } else {
                rhs[vid] -= bv(i as isize, j as isize - 1);
            }

            if i < rows - 1 {
                let tid = (i + 1) * cols + j;
                ret_a.add_triplet(vid, tid, 1.);
            } else {
                rhs[vid] -= bv(i as isize + 1, j as isize);
            }

            if j < cols - 1 {
                let tid = i * cols + j + 1;
                ret_a.add_triplet(vid, tid, 1.);
            } else {
                rhs[vid] -= bv(i as isize, j as isize + 1);
            }
        } // end for
    } // end for
    (ret_a.to_csr(), rhs)
}
