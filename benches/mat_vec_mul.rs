use criterion::{criterion_group, criterion_main, Criterion};
use sprsolve::MatVecMul;

#[cfg(feature = "parallel")]
fn set_threads() {
    // Consider setting a fixed number of threads here, for example to avoid
    // oversubscribing on hyperthreaded cores.
    let n = 4;
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global();
    println!("BENCH with {} threads", n);
}

fn mat_vec_mul(c: &mut Criterion) {
    #[cfg(feature = "parallel")]
    set_threads();

    let res = 100;
    let (rows, cols) = (res, res);
    let lap = grid_laplacian((rows, cols));
    let mut rhs = vec![0_f64; rows * cols];
    set_boundary_condition(rhs.as_mut_slice(), (rows, cols), |row, col| {
        (row + col) as f64
    });

    let mut x = vec![0_f64; rows * cols];

    let name = format!("Laplacian-Mul-{}", res);
    c.bench_function(&name, |b| {
        b.iter(|| unsafe {
            lap.mul_vec_unchecked(rhs.as_slice(), x.as_mut_slice());
        })
    });
}

criterion_group!(benches, mat_vec_mul);
criterion_main!(benches);

// --------------------------------------------------------------------------------

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
