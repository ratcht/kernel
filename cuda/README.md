# CUDA

Hand-written CUDA kernel implementations for common operations. Each operation has multiple variants demonstrating different optimization techniques.

## Structure

```
cuda/
├── kernels/           # kernel implementations (.cuh files)
├── benchmarks/        # benchmark programs to run and time kernels
├── utils/             # helper functions (printing, validation, timing)
└── Makefile           # build all benchmarks
```

## Operations

**vector_add**
- `naive`: basic implementation with explicit memory management
- `unified_memory`: uses cudaMallocManaged

**matmul**
- `naive`: basic implementation for MxN matrices
- `naive_square`: optimized for square matrices
- `coalesced`: transposes matrix A for coalesced memory access
- `tiled`: shared memory tiling

**conv1d**
- `naive`: basic 1D convolution
- `constant_memory`: stores kernel in constant memory
- `tiled`: uses shared memory for input
- `tiled_padded`: handles padding with shared memory
- `strided_padded`: supports stride and padding parameters

**conv2d**
- `tiled`: 2D convolution with shared memory and constant memory for kernel

**reduction**
- `divergent`: warp divergence due to modulo operation
- `bank_conflicts`: sequential threads but causes bank conflicts
- `no_bank_conflicts`: reversed stride to avoid conflicts
- `reduce_idle_threads`: first add during load to reduce idle threads
- `first_add_during_load`: multiple adds per thread with grid-stride loop
- `unroll_last_warp`: manually unroll last 32 iterations
- `completely_unrolled`: fully unrolled reduction
- `cooperative_groups`: uses cooperative groups API

**softmax**
- `naive`: each thread processes one row independently
- `shared_memory`: uses shared memory for row-wise reductions
- `warp_shuffle`: uses warp shuffle intrinsics for faster reduction

## Building

Compile all benchmarks:
```bash
make
```

Compile specific benchmark:
```bash
make bench_matmul
```

Clean binaries:
```bash
make clean
```

## Running Benchmarks

Each benchmark takes a kernel number and size parameters.

**Vector Add**
```bash
./benchmarks/bench_vector_add 0  # naive
./benchmarks/bench_vector_add 1  # unified memory
```

**Matrix Multiplication**
```bash
./benchmarks/bench_matmul 0 128    # naive, 128x128 matrix
./benchmarks/bench_matmul 2 1024   # tiled, 1024x1024 matrix
```

**Conv1d**
```bash
./benchmarks/bench_conv1d 0 1024 3      # naive, input_size=1024, kernel_size=3
./benchmarks/bench_conv1d 2 4096        # tiled (uses kernel_size=4)
./benchmarks/bench_conv1d 4 14 2 4 4    # strided_padded with custom params
```

**Conv2d**
```bash
./benchmarks/bench_conv2d 64 64         # 64x64 input, default 2x2 kernel
./benchmarks/bench_conv2d 128 128 3 3 1 # 128x128 input, 3x3 kernel, padding=1
```

**Reduction**
```bash
./benchmarks/bench_reduction 0 64       # divergent, array_size=64
./benchmarks/bench_reduction 7 1048576  # cooperative groups, array_size=1M
```

**Softmax**
```bash
./benchmarks/bench_softmax 0 8 8        # naive, 8x8 matrix
./benchmarks/bench_softmax 2 1024 1024  # warp shuffle, 1024x1024 matrix
```
