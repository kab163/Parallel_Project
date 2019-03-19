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

__global__ void matrixMult(const int N, int part, float *d_A, float *d_B, float *d_C) 
{
  int index = threadIdx.x + blockIdx.x * blockDim.x; 
  double pSum = 0.0;

  if (index < part * N) 
  { 
    int r = index / N;
    int c = index % N;

#pragma loop unroll
    for (int i = 0; i < N; i++) {
      pSum += d_A[r * N + i] * d_B[i * N + c];
    }
    d_C[index] = pSum; 
  } 
}

int main(int argc, char* argv[])
{
  if(argc != 3) {fprintf(stderr, "usage: <exe>, size_of_array, num_groups\n"); exit(-1);}

  const int N = atoi(argv[1]);
  float *d_A, *d_B, *d_C; //device variables 

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Running on %s\n", prop.name); 

  //timing
  struct timeval start, end; 
  double runtime = 0.0, total = 0.0;

  const int factor = atoi(argv[2]);
  int part = N / factor;

  //allocate on CPU
  float *arrayA, *arrayB, *arrayC;
  cudaHostAlloc(&arrayA, N * N * sizeof(float), cudaHostAllocDefault); CudaTest("failed allocation");
  cudaHostAlloc(&arrayB, N * N * sizeof(float), cudaHostAllocDefault); CudaTest("failed allocation");
  cudaHostAlloc(&arrayC, N * N * sizeof(float), cudaHostAllocDefault); CudaTest("failed allocation");

  //create cuda streams
  cudaStream_t streams[factor];
  for(int i = 0; i < factor; i++) {cudaStreamCreate(&streams[i]);}

  //fill array inputs on CPU
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {     
      arrayA[i * N + j] = 1.0 * (i * N + j);
      arrayB[i * N + j] = 2.0 * i;
    } 

  //allocate space on GPU
  cudaMalloc((void**) &d_A, part * N * sizeof(float)); CudaTest("failed allocation");
  cudaMalloc((void**) &d_B, N * N * sizeof(float)); CudaTest("failed allocation");
  cudaMalloc((void**) &d_C, part * N * sizeof(float)); CudaTest("failed allocation");

  cudaMemcpyAsync(d_B, arrayB, N * N * sizeof(float), cudaMemcpyHostToDevice); CudaTest("failed to send data to GPU");

  for(int i = 0; i < factor; i++) {
    //send part data to GPU for MM
    cudaMemcpyAsync(d_A, &arrayA[(i * part) * N], part * N * sizeof(float), cudaMemcpyHostToDevice, streams[i]); 
                                                                  CudaTest("failed to send data to GPU");
 
    //run first kernel
    gettimeofday(&start, NULL);   
    matrixMult<<<((part * N + THREADS -1)/THREADS), THREADS, 0, streams[i]>>> (N, part, d_A, d_B, d_C); //CudaTest("failed kernel"); 
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    runtime = end.tv_sec + (end.tv_usec / 1000000.0) - start.tv_sec - (start.tv_usec / 1000000.0);
    total += runtime;
   
    //send part data back to CPU for MM
    cudaMemcpyAsync(&arrayC[(i * part) * N], d_C, part * N * sizeof(float), cudaMemcpyDeviceToHost, streams[i]); 
                                                                         CudaTest("failed to send data back");
  }
  printf("\nCompute time for Matrix Multiply: %.8f s\n", total);

  //check result
/*  for (int i = 0; i < N; i++) 
    for(int j = 0; j < N; j++) 
    { 
      printf("Array C: %.2lf \n", arrayC[i * N + j]);  
    }
*/

  //free streams
  for(int i = 0; i < factor; i ++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  //free memory
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  cudaFreeHost(arrayA); cudaFreeHost(arrayB); cudaFreeHost(arrayC); 
  
  return 0;
}
