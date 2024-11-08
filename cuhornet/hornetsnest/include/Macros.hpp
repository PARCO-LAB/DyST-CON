
// Useful for grid strided work
#define KERNEL_FOR_STRIDED(var, N)                                             \
  for (var = blockDim.x * blockIdx.x + threadIdx.x; var < N;                   \
       var += blockDim.x * gridDim.x)

// Useful for standard thread/work usage
#define BLOCK_COUNT(size, work) ((size + work - 1) / work)

// Useful with timers
#define INSTRUMENT(timer, result, expr)                                        \
  {                                                                            \
    timer.start();                                                             \
    expr;                                                                      \
    timer.stop();                                                              \
    result = timer.duration();                                                 \
  }
