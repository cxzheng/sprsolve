// large part of this test code comes from (https://github.com/vbarrielle/sprs/blob/master/examples/heat.rs)
#[test]
fn test_gauss_seidel() {
    let (rows, cols) = (10, 10);
    let lap = grid_laplacian((rows, cols));
    println!(
        "grid laplacian nnz structure:\n{}",
        sprs::visu::nnz_pattern_formatter(lap.view()),
    );
    let mut rhs = vec![0_f64; rows * cols];
    set_boundary_condition(rhs.as_mut_slice(), (rows, cols), |row, col| {
        (row + col) as f64
    });

    let mut x = vec![0_f64; rows * cols];
    let mut solver = sprsolve::GaussSeidel::new(lap.view()).unwrap();
    let (iters, res) = solver
        .solve(rhs.as_slice(), x.as_mut_slice(), 300, 0.)
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
fn test_bicg_stab() {
    let (rows, cols) = (20, 20);
    let lap = grid_laplacian((rows, cols));
    let mut rhs = vec![0_f64; rows * cols];
    set_boundary_condition(rhs.as_mut_slice(), (rows, cols), |row, col| {
        (row + col) as f64
    });

    let mut x = vec![0_f64; rows * cols];
    let mut solver = sprsolve::BiCGStab::new(&lap, lap.cols());
    let (iters, res) = solver
        .solve(rhs.as_slice(), x.as_mut_slice(), 1500, 1E-17)
        .unwrap();
    for i in 0..rows {
        for j in 0..cols {
            print!("{} ", x[i * rows + j]);
        }
        println!("");
    }
    println!(
        "Solved system in {} iterations with relative residual error {}",
        iters, res
    );
}

/// Determine whether the grid location at `(row, col)` is a border
/// of the grid defined by `shape`.
fn is_border(row: usize, col: usize, shape: (usize, usize)) -> bool {
    let (rows, cols) = shape;
    let top_row = row == 0;
    let bottom_row = row + 1 == rows;
    let border_row = top_row || bottom_row;

    let left_col = col == 0;
    let right_col = col + 1 == cols;
    let border_col = left_col || right_col;

    border_row || border_col
}

fn grid_laplacian(shape: (usize, usize)) -> sprs::CsMat<f64> {
    let (rows, cols) = shape;
    let nb_vert = rows * cols;
    let mut indptr = Vec::with_capacity(nb_vert + 1);
    let nnz = 5 * nb_vert + 5;
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    let mut cumsum = 0;

    for i in 0..rows {
        for j in 0..cols {
            indptr.push(cumsum);

            let mut add_elt = |i, j, x| {
                indices.push(i * rows + j);
                data.push(x);
                cumsum += 1;
            };

            if is_border(i, j, shape) {
                // establish Dirichlet boundary conditions
                add_elt(i, j, 1.);
            } else {
                add_elt(i - 1, j, 1.);
                add_elt(i, j - 1, 1.);
                add_elt(i, j, -4.);
                add_elt(i, j + 1, 1.);
                add_elt(i + 1, j, 1.);
            }
        }
    }

    indptr.push(cumsum);

    sprs::CsMat::new((nb_vert, nb_vert), indptr, indices, data)
}

fn set_boundary_condition<F>(rhs: &mut [f64], grid_shape: (usize, usize), f: F)
where
    F: Fn(usize, usize) -> f64,
{
    let (rows, cols) = grid_shape;
    for i in 0..rows {
        for j in 0..cols {
            if is_border(i, j, grid_shape) {
                let index = i * rows + j;
                rhs[index] = f(i, j);
            }
        }
    }
}
