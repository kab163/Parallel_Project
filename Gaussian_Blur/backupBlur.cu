#include <cstdio>
#include <iostream>
#include <stdlib.h>

#define cimg_use_jpeg
#include "CImg.h"

#define THREADS 128

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


static __global__ void blur(unsigned char *d_rout, unsigned char *d_gout, unsigned char *d_bout, unsigned char *d_rO, unsigned char *d_gO, unsigned char *d_bO, const int H, const int W) {

  int index= (threadIdx.x) + blockIdx.x * blockDim.x;
  int r,c; 
  //compute the blur
  if (index > 0 && index < (H) * (W)) {
    r = index / W;
    c = index % W;

    //red
    d_rout[index] = (d_rO[(r+1) * W + c] + d_rO[(r-1) * W + c] +
                           d_rO[r * W + (c+1)] + d_rO[(r+1) * W + (c+1)] + d_rO[(r-1) * W + (c+1)] +
                           d_rO[r * W + (c-1)] + d_rO[(r+1) * W + (c-1)] + d_rO[(r-1) * W + (c-1)]) / 8;

    //green
    d_gout[index] = (d_gO[(r+1) * W + c] + d_gO[(r-1) * W + c] +
                           d_gO[r * W + (c+1)] + d_gO[(r+1) * W + (c+1)] + d_gO[(r-1) * W + (c+1)] +
                           d_gO[r * W + (c-1)] + d_gO[(r+1) * W + (c-1)] + d_gO[(r-1) * W + (c-1)]) / 8;

    //blue
    d_bout[index] = (d_bO[(r+1) * W + c] + d_bO[(r-1) * W + c] +
                           d_bO[r * W +(c+1)] + d_bO[(r+1) * W + (c+1)] + d_bO[(r-1) * W + (c+1)] +
                           d_bO[r * W + (c-1)] + d_bO[(r+1) * W + (c-1)] + d_bO[(r-1) * W + (c-1)]) / 8;
  }
}

int main(int argc, char *argv[]) {
  struct timeval start, end;
  double runtime = 0.0, total = 0.0;

  if (argc != 3) {
    fprintf(stderr, "usage: exe, input file, number of groups\n"); exit(-1);
  }
  
  // import image from jpg file
  cimg_library::CImg<unsigned char> input_img(argv[1]);

  const int NUM_PART = atoi(argv[2]);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Running on %s\n", prop.name);

  //create Height/Width variables for readability
  const int H = input_img.height();
  const int W = input_img.width();
  const int Hout = H + 2;
  const int Wout = W + 2;

  //create height/offset for partitions
  int offsetH = H / NUM_PART; //height for each group, not padded

  //create GPU variables
  unsigned char *d_rO, *d_gO, *d_bO, *d_rout, *d_gout, *d_bout;

  //allocate CPU arrays
  unsigned char* rO = (unsigned char*)calloc(Wout * Hout, sizeof(unsigned char));
  unsigned char* gO = (unsigned char*)calloc(Wout * Hout, sizeof(unsigned char));
  unsigned char* bO = (unsigned char*)calloc(Wout * Hout, sizeof(unsigned char));
  unsigned char* rout = (unsigned char*)malloc(W * H * sizeof(unsigned char));
  unsigned char* gout = (unsigned char*)malloc(W * H * sizeof(unsigned char));
  unsigned char* bout = (unsigned char*)malloc(W * H * sizeof(unsigned char)); 

  //allocate GPU memory for arrays
  cudaMalloc((void**)&d_rO, Wout * (offsetH+2) * sizeof(unsigned char));
  cudaMalloc((void**)&d_gO, Wout * (offsetH+2) * sizeof(unsigned char));
  cudaMalloc((void**)&d_bO, Wout * (offsetH+2) * sizeof(unsigned char));
  cudaMalloc((void**)&d_rout, Wout * (offsetH+2) * sizeof(unsigned char));
  cudaMalloc((void**)&d_gout, Wout * (offsetH+2) * sizeof(unsigned char));
  cudaMalloc((void**)&d_bout, Wout * (offsetH+2) * sizeof(unsigned char));

  //pad the image
  for(int c = 0; c< W; c++) {
    for(int r = 0; r < H; r++) {
      rO[(r+1) * W + (c+1) ] = input_img(c, r, 0);
      gO[(r+1) * W + (c+1) ] = input_img(c, r, 1);
      bO[(r+1) * W + (c+1) ] = input_img(c, r, 2);
    }
  }

  //create new image   
  cimg_library::CImg<unsigned char> output_img(W, H, 1, 3);

  //loop over number of groups, calculate portion of blur for each
  for(int i = 0; i < NUM_PART; i++) {
    //send over padded image info to GPU
    if(cudaSuccess != cudaMemcpy(d_rO, &rO[(i * (offsetH)) * Wout], Wout * (offsetH+2) * sizeof(unsigned char), cudaMemcpyHostToDevice)) fprintf(stderr, "copy to device failed\n");
    if(cudaSuccess != cudaMemcpy(d_gO, &gO[(i * (offsetH)) * Wout], Wout * (offsetH+2) * sizeof(unsigned char), cudaMemcpyHostToDevice)) fprintf(stderr, "copy to device failed\n");
    if(cudaSuccess != cudaMemcpy(d_bO, &bO[(i * (offsetH)) * Wout], Wout * (offsetH+2) * sizeof(unsigned char), cudaMemcpyHostToDevice)) fprintf(stderr, "copy to device failed\n");

    //launch kernel
    gettimeofday(&start, NULL);
    blur<<<((offsetH+1) * (W+1) + THREADS - 1) / THREADS, THREADS>>>(d_rout, d_gout, d_bout, d_rO, d_gO, d_bO, (offsetH+1), (W+1));
    gettimeofday(&end, NULL);

    runtime = end.tv_sec + (end.tv_usec / 1000000.0) - start.tv_sec - (start.tv_usec / 1000000.0);
    total+= runtime;

    //send the blurred image info back to CPU
    if(cudaSuccess != cudaMemcpy(&rout[((i * (offsetH)) * Wout)], d_rout, W * (offsetH) * sizeof(unsigned char), cudaMemcpyDeviceToHost)) fprintf(stderr, "copy to host failed\n");
    if(cudaSuccess != cudaMemcpy(&gout[((i * (offsetH)) * Wout)], d_gout, W * (offsetH) * sizeof(unsigned char), cudaMemcpyDeviceToHost)) fprintf(stderr, "copy to host failed\n");
    if(cudaSuccess != cudaMemcpy(&bout[((i * (offsetH)) * Wout)], d_bout, W * (offsetH) * sizeof(unsigned char), cudaMemcpyDeviceToHost)) fprintf(stderr, "copy to host failed\n");
  }

  printf("\nCompute time for Blur: %.8f s\n", runtime);

  for(int c = 0; c < W; c++) {
    for(int r = 0; r < H; r++) {
      output_img(c, r, 0) = rout[(r)* W + (c)];
      output_img(c, r, 1) = gout[(r)* W + (c)];
      output_img(c, r, 2) = bout[(r)* W + (c)];
    }
  }
 
  //save output to file
  output_img.save_jpeg("output.jpg");

  cudaFree(d_rO); cudaFree(d_gO); cudaFree(d_bO);
  cudaFree(d_rout); cudaFree(d_gout); cudaFree(d_bout);
  free(rO); free(gO); free(bO); free(rout); free(gout); free(bout);
  return 0;
}
