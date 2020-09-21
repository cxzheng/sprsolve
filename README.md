# sprsolve
Rust implementation of various sparse linear solvers, with pedagogical and research purposes in mind.
Still working in progress ...

## Performance

When using MKL (by enabling _mkl_ feature), test to use the features between 
_mkl-static-lp64-iomp_ and _mkl-static-lp64-seq_.

When using rayon (by enabling _parallel_ feature), test the number of threads enabled.

-Benchmark
```
    cargo bench --bench bench_bicg_stab
```

When MKL with iomp is enabled, you might want to use
```
RUSTFLAGS="-L /opt/intel/lib/intel64" cargo bench --bench mkl_bicgstab
```
