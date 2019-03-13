#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>

const int THREADS = 512;

static void CudaTest(const char *msg)
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    exit(-1);
  }
}

__global__ void matrixMult(const int N, double *d_A, double *d_B, double *d_C) 
{
  int index = threadIdx.x + blockIdx.x * blockDim.x; 
  double pSum = 0.0;

  if (index < N * N) 
  { 
    int r = index / N;
    int c = index % N;

    for (int i = 0; i < N; i++) {
      pSum += d_A[r * N + i] * d_B[i * N + c];
    }
    d_C[index] = pSum; 
  } 
}

int main(int argc, char* argv[])
{
  if(argc != 2) {fprintf(stderr, "usage: <exe>, size_of_array\n"); exit(-1);}

  const int N = atoi(argv[1]);
  double *d_A, *d_B, *d_C; //device variables 
  
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Running on %s\n", prop.name);

  //timing
  struct timeval start, end;

  //allocate on CPU
  double *arrayA = (double *)malloc(N * N * sizeof(double));
  double *arrayB = (double *)malloc(N * N * sizeof(double));
  double *arrayC = (double *)malloc(N * N * sizeof(double)); 

  //fill array inputs on CPU
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) { 
      arrayA[i * N + j] = (i * N + j) + rand() % 100000 / 100000.0;
      arrayB[i * N + j] = 2 * rand() % 100000 / 100000.0;
    }

  //allocate space on GPU
  cudaMalloc((void**) &d_A, N * N * sizeof(double)); CudaTest("failed allocation");
  cudaMalloc((void**) &d_B, N * N * sizeof(double)); CudaTest("failed allocation");
  cudaMalloc((void**) &d_C, N * N * sizeof(double)); CudaTest("failed allocation");

  //send data to GPU
  cudaMemcpy(d_A, arrayA, N * N * sizeof(double), cudaMemcpyHostToDevice); CudaTest("failed to send data to GPU");
  cudaMemcpy(d_B, arrayB, N * N * sizeof(double), cudaMemcpyHostToDevice); CudaTest("failed to send data to GPU2");
 
  //run and time kernel
  gettimeofday(&start, NULL);
  matrixMult<<<((N * N + THREADS -1)/THREADS), THREADS>>> (N, d_A, d_B, d_C); //CudaTest("failed kernel");
  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + (end.tv_usec / 1000000.0) - start.tv_sec - (start.tv_usec / 1000000.0);
  printf("\nCompute time for Matrix Multiply: %.8f s\n", runtime);

  //send data back to CPU
  cudaMemcpy(arrayC, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost); CudaTest("failed to send data back");

   
  //check result
/*  for (int i = 0; i < N; i++) 
    for(int j = 0; j < N; j++) 
    { 
      printf("Array C: %.2lf \n", arrayC[i * N + j]);  
    }
*/

  //free memory
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  free(arrayA); free(arrayB); free(arrayC);
  return 0;
}
