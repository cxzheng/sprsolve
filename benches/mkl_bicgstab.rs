#[cfg(feature = "mkl")]
use sprsolve::MklMat;
use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(not(feature = "mkl"))]
fn bench_mkl_bicgstab(_c: &mut Criterion) {
}

#[cfg(feature = "mkl")]
fn bench_mkl_bicgstab(c: &mut Criterion) {
    let res = 100;
    let (rows, cols) = (res, res);
    let lap = grid_laplacian((rows, cols));
    let mut rhs = vec![0_f64; rows * cols];
    set_boundary_condition(rhs.as_mut_slice(), (rows, cols), |row, col| {
        (row + col) as f64
    });

    let mkl_mat = MklMat::new(lap).unwrap();

    let mut x = vec![0_f64; rows * cols];
    mkl_mat.mv_and_dotmv_hint(1500).unwrap();
    let mut solver = sprsolve::BiCGStab::new(&mkl_mat, mkl_mat.size());

    let name = format!("(MKL) Laplacian-{}", res);
    c.bench_function(&name, |b| {
        b.iter(|| {
            solver
                .solve(rhs.as_slice(), x.as_mut_slice(), 1500, 1E-16)
                .unwrap();
        })
    });
}

/// Determine whether the grid location at `(row, col)` is a border
/// of the grid defined by `shape`.
#[allow(dead_code)]
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

#[allow(dead_code)]
fn grid_laplacian(shape: (usize, usize)) -> sprs::CsMatI<f64, i32> {
    let (rows, cols) = shape;
    let nb_vert = rows * cols;
    let mut indptr: Vec<i32> = Vec::with_capacity(nb_vert + 1);
    let nnz = 5 * nb_vert + 5;
    let mut indices: Vec<i32> = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    let mut cumsum = 0;

    for i in 0..rows {
        for j in 0..cols {
            indptr.push(cumsum);

            let mut add_elt = |i, j, x| {
                indices.push((i * rows + j) as i32);
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

    sprs::CsMatI::new((nb_vert, nb_vert), indptr, indices, data)
}

#[allow(dead_code)]
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

criterion_group!(benches, bench_mkl_bicgstab);
criterion_main!(benches);
