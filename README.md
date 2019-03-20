# Parallel_Project
Final Project for CIS531

_____________________________________

Matrix Multiply:

To build: run "make"

To run optimized version: ./fast [size of matrix] [number of partitions]

To run normal GPU version: ./single [size of matrix]

The normal GPU version of the code will send the entire matrix to the GPU. The optimized version will break up matrix A into partitions, and send the appropriate row of matrix B. The resulting matrices for both versions will be in C. The optimized version has several optimizations such as (1) using floats instead of doubles, (2) using asynchronous memory transfers, (3) compiler optimizations, and (4) using CUDA streams. We can see a speedup of the optimizied version over the normal GPU version, confirming that these are worthwhile improvements - see report for a more thorough analysis.

_____________________________________

Fractal:

To build: run "make"

To run optimized version: ./fast [width] [depth] [number of partitions]

To run normal GPU version ./single [width] [depth]

The normal GPU version will compute a an entire Mandelbrot fractal. The optimized version will break up the fractal into chunks, and send each chunk to the GPU to be calculated. We do see that the partitioned version can compute much much larger fractals, since they can be broken up. The GPU memory size is no longer a limit for the input. Some optimizations for the partitioned version include those mentioned above, except I do not use CUDA streams here. I also use unsigned char, not floats.

_____________________________________

nBody:

To build: run "make"

To run optimized version: ./fast [number of bodies] [number of timesteps] [number of partitions]

To run normal GPU version: ./single [number of bodies] [number of timesteps]

The normal GPU version will compute the evolution of the positions of the bodies for each time step. The optimized version will break up the total number of bodies into the specified number of chunks. It will then send that chunk of data to the GPU to be computed. It is difficult to get speed up with this implementation because the nbody algorithm is compute-bound, and therefore a favorable computation for the GPU. When the input size is broken up, it's hard to have the GPU continue to be fully loaded and thus get good performance. This kind of regular algorithm may require special care to optimize. We see performance improvement given the right conditions - see the report for a more thorough analysis.

____________________________________

*Note: I decided later on not to use the Gaussian Blur code. I left the code where I left off in this repository for future work purposes.*_
