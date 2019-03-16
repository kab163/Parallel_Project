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
  if (index < (H) * (W)) {
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
	
  if (argc != 2) {
    fprintf(stderr, "usage: exe, input file\n"); exit(-1);
  }
  
  // import image from jpg file
  cimg_library::CImg<unsigned char> input_img(argv[1]); 

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Running on %s\n", prop.name);

  //create Height/Width variables for readability
  const int H = input_img.height();
  const int W = input_img.width();
  const int Hout = H + 2;
  const int Wout = W + 2;

  //create GPU variables
  unsigned char *d_rO, *d_gO, *d_bO, *d_rout, *d_gout, *d_bout;

  //allocate CPU arrays
  unsigned char* rO = (unsigned char*)calloc(Wout * Hout, sizeof(unsigned char));
  unsigned char* gO = (unsigned char*)calloc(Wout * Hout, sizeof(unsigned char));
  unsigned char* bO = (unsigned char*)calloc(Wout * Hout, sizeof(unsigned char));
  unsigned char* rout = (unsigned char*)calloc(Wout * Hout, sizeof(unsigned char));
  unsigned char* gout = (unsigned char*)calloc(Wout * Hout, sizeof(unsigned char));
  unsigned char* bout = (unsigned char*)calloc(Wout * Hout, sizeof(unsigned char)); 

  //allocate GPU memory for arrays
  cudaMalloc((void**)&d_rO, Wout * Hout * sizeof(unsigned char));
  cudaMalloc((void**)&d_gO, Wout * Hout * sizeof(unsigned char));
  cudaMalloc((void**)&d_bO, Wout * Hout * sizeof(unsigned char));
  cudaMalloc((void**)&d_rout, Wout * Hout * sizeof(unsigned char));
  cudaMalloc((void**)&d_gout, Wout * Hout * sizeof(unsigned char));
  cudaMalloc((void**)&d_bout, Wout * Hout * sizeof(unsigned char));


  for(int c = 0; c< W; c++) {
    for(int r = 0; r < H; r++) {

      rO[(r+1)*W + (c+1) ] = input_img(c, r, 0);
      gO[(r+1)*W + (c+1) ] = input_img(c, r, 1);
      bO[(r+1)*W + (c+1) ] = input_img(c, r, 2);

    }
  }

  //create new image   
  cimg_library::CImg<unsigned char> output_img(W, H, 1, 3);

  //send over padded image info to GPU
  if(cudaSuccess != cudaMemcpy(d_rO, rO, Wout * Hout * sizeof(unsigned char), cudaMemcpyHostToDevice)) fprintf(stderr, "copy to device failed\n");
  if(cudaSuccess != cudaMemcpy(d_gO, gO, Wout * Hout * sizeof(unsigned char), cudaMemcpyHostToDevice)) fprintf(stderr, "copy to device failed\n");
  if(cudaSuccess != cudaMemcpy(d_bO, bO, Wout * Hout * sizeof(unsigned char), cudaMemcpyHostToDevice)) fprintf(stderr, "copy to device failed\n");

  //launch kernel
  gettimeofday(&start, NULL);
  blur<<<(Hout * Wout + THREADS - 1) / THREADS, THREADS>>>(d_rout, d_gout, d_bout, d_rO, d_gO, d_bO, Hout, Wout);  
  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);

  double runtime = end.tv_sec + (end.tv_usec / 1000000.0) - start.tv_sec - (start.tv_usec / 1000000.0);
  printf("\nCompute time for Blur: %.8f s\n", runtime);

  //send the blurred image info back to CPU
  if(cudaSuccess != cudaMemcpy(rout, d_rout, Wout * Hout * sizeof(unsigned char), cudaMemcpyDeviceToHost)) fprintf(stderr, "copy to host failed\n");
  if(cudaSuccess != cudaMemcpy(gout, d_gout, Wout * Hout * sizeof(unsigned char), cudaMemcpyDeviceToHost)) fprintf(stderr, "copy to host failed\n");
  if(cudaSuccess != cudaMemcpy(bout, d_bout, Wout * Hout * sizeof(unsigned char), cudaMemcpyDeviceToHost)) fprintf(stderr, "copy to host failed\n");


  for(int c = 0; c < W; c++) {
    for(int r = 0; r < H; r++) {
      output_img(c, r, 0) = rout[(r+1)*W + (c+1)];
      output_img(c, r, 1) = gout[(r+1)*W + (c+1)];
      output_img(c, r, 2) = bout[(r+1)*W + (c+1)];
    }
  }

  /*
  //Check output is exact as input
  for(int c = 0; c < W; c++) {
    for(int r = 0; r < H; r++) {

	    if (output_img(c, r, 0) != input_img(c,r,0))
		printf("red c: %d r:%d\n",c,r);    
	    if (output_img(c, r, 1) != input_img(c,r,1))
		printf("green c: %d r:%d\n",c,r);    
	    if (output_img(c, r, 2) != input_img(c,r,2))
		printf("blue c: %d r:%d\n",c,r);    
                //printf("out: %d : input: %d, \n", output_img(c, r, 0), input_img(c,r,0));
      

    }
  }
*/
  //save output to file
  output_img.save_jpeg("output.jpg");

  cudaFree(d_rO); cudaFree(d_gO); cudaFree(d_bO);
  cudaFree(d_rout); cudaFree(d_gout); cudaFree(d_bout);
  free(rO); free(gO); free(bO); free(rout); free(gout); free(bout);
  return 0;
}
