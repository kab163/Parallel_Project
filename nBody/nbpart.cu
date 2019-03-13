#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>


#define THREADS 128

__constant__ int nbodiesd;
__constant__ float dthfd, epssqd;
__constant__ float *massd, *posxd, *posyd, *poszd, *velxd, *velyd, *velzd, *accxd, *accyd, *acczd;


/******************************************************************************/

// input generation (follows SPLASH2)

#define MULT 1103515245
#define ADD 12345
#define MASK (0x7FFFFFFF)
#define TWOTO31 2147483648.0

static int A = 1;
static int B = 0;
static int randx = 1;
static int lastrand;

static void drndset(int seed)
{
   A = 1;
   B = 0;
   randx = (A*seed+B) & MASK;
   A = (MULT * A) & MASK;
   B = (MULT*B + ADD) & MASK;
}

static double drnd()
{
   lastrand = randx;
   randx = (A*randx+B) & MASK;
   return((double)lastrand/TWOTO31);
}

static void generateInput(int nbodies, float *mass, float *posx, float *posy, float *posz, float *velx, float *vely, float *velz)
{
  register int i;
  register double rsc, vsc, r, v, x, y, z, sq, scale;

  drndset(7);
  rsc = (3 * 3.1415926535897932384626433832795) / 16;
  vsc = sqrt(1.0 / rsc);
  for (i = 0; i < nbodies; i++) {
    mass[i] = 1.0 / nbodies;
    r = 1.0 / sqrt(pow(drnd()*0.999, -2.0/3.0) - 1);
    do {
      x = drnd()*2.0 - 1.0;
      y = drnd()*2.0 - 1.0;
      z = drnd()*2.0 - 1.0;
      sq = x*x + y*y + z*z;
    } while (sq > 1.0);
    scale = rsc * r / sqrt(sq);
    posx[i] = x * scale;
    posy[i] = y * scale;
    posz[i] = z * scale;

    do {
      x = drnd();
      y = drnd() * 0.1;
    } while (y > x*x * pow(1 - x*x, 3.5));
    v = x * sqrt(2.0 / sqrt(1 + r*r));
    do {
      x = drnd()*2.0 - 1.0;
      y = drnd()*2.0 - 1.0;
      z = drnd()*2.0 - 1.0;
      sq = x*x + y*y + z*z;
    } while (sq > 1.0);
    scale = vsc * v / sqrt(sq);
    velx[i] = x * scale;
    vely[i] = y * scale;
    velz[i] = z * scale;
  }
}


/******************************************************************************/
/*** compute force ************************************************************/
/******************************************************************************/

__global__ __launch_bounds__(THREADS, 2) void ForceCalculationKernel(int height, int chunk, int step)
{
  register int i, j1, j2, id, bound, idx;
  register float px, py, pz, ax, ay, az, dx, dy, dz, tmp;
  __shared__ float posxs[THREADS], posys[THREADS], poszs[THREADS], masss[THREADS];

  id = threadIdx.x;
  i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < height) {
    px = posxd[chunk + i];
    py = posyd[chunk + i];
    pz = poszd[chunk + i];

    ax = 0.0f;
    ay = 0.0f;
    az = 0.0f;
  

    for (j1 = 0; j1 < nbodiesd; j1 += THREADS) {
      idx = id + j1;
      if (idx < nbodiesd) {
        posxs[id] = posxd[idx];
        posys[id] = posyd[idx];
        poszs[id] = poszd[idx];
        masss[id] = massd[idx];
      }
      __syncthreads();
 
      bound = min(nbodiesd - j1, THREADS);
      if (bound == THREADS) {
        #pragma unroll
        for (j2 = 0; j2 < THREADS; j2++) {
          dx = posxs[j2] - px;
          dy = posys[j2] - py;
          dz = poszs[j2] - pz;
          tmp = dx*dx + (dy*dy + (dz*dz + epssqd));
          tmp = rsqrtf(tmp);
          tmp = masss[j2] * tmp * tmp * tmp;
          ax += dx * tmp;
          ay += dy * tmp;
          az += dz * tmp;
        }
      } else {
        for (j2 = 0; j2 < bound; j2++) {
          dx = posxs[j2] - px;
          dy = posys[j2] - py;
          dz = poszs[j2] - pz;
          tmp = dx*dx + (dy*dy + (dz*dz + epssqd));
          tmp = rsqrtf(tmp);
          tmp = masss[j2] * tmp * tmp * tmp;
          ax += dx * tmp;
          ay += dy * tmp;
          az += dz * tmp;
        }
      }
   
      __syncthreads();
    }

 
    if (step > 0) {
      velxd[i] += (ax - accxd[i]) * dthfd;
      velyd[i] += (ay - accyd[i]) * dthfd;
      velzd[i] += (az - acczd[i]) * dthfd;
    }

    accxd[i] = ax;
    accyd[i] = ay;
    acczd[i] = az;
  }
}


/******************************************************************************/
/*** advance bodies ***********************************************************/
/******************************************************************************/

__global__ __launch_bounds__(THREADS, 2) void IntegrationKernel(int height, int chunk)
{
  register int i;
  register float dtime;
  register float dvelx, dvely, dvelz;
  register float velhx, velhy, velhz;

  i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < height) {
    dtime = dthfd * 2.0f;

    dvelx = accxd[i] * dthfd;
    dvely = accyd[i] * dthfd;
    dvelz = acczd[i] * dthfd;

    velhx = velxd[i] + dvelx;
    velhy = velyd[i] + dvely;
    velhz = velzd[i] + dvelz;

    posxd[chunk + i] += velhx * dtime;
    posyd[chunk + i] += velhy * dtime;
    poszd[chunk + i] += velhz * dtime;

    velxd[i] = velhx + dvelx;
    velyd[i] = velhy + dvely;
    velzd[i] = velhz + dvelz;
  }
}


/******************************************************************************/

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


/******************************************************************************/

int main(int argc, char *argv[])
{
  register int blocks, timesteps, groups;
  float runtime; 
  struct timeval starttime, endtime;

  register int nbodies, step, nbparts;
  register float dthf, epssq;
  register float *mass, *posx, *posy, *posz, *velx, *vely, *velz;
  register float *massl, *posxl, *posyl, *poszl, *velxl, *velyl, *velzl, *accxl, *accyl, *acczl;

  printf("n-body GPU Kepler\n");
  if (argc != 4) {fprintf(stderr, "\narguments: number_of_bodies number_of_timesteps num_groups\n"); exit(-1);}

  nbodies = atoi(argv[1]);
  timesteps = atoi(argv[2]);
  groups = atoi(argv[3]);
  dthf = 0.025f * 0.5f;
  epssq = 0.05f * 0.05f;

  cudaStream_t streams[groups];
  for(int i = 0; i < groups; i++) {cudaStreamCreate(&streams[i]);}

  if(nbodies % groups != 0) {fprintf(stderr, "Group number must evenly divide number of bodies\n"); exit(-1);}
  nbparts = nbodies / groups; //number of bodies in each group.

  if (nbodies < 1) {fprintf(stderr, "nbodies is too small: %d\n", nbodies); exit(-1);} 
  printf("configuration: %d bodies, %d time steps, %d groups \n", nbodies, timesteps, groups);

  cudaHostAlloc((void**) &mass, (sizeof(float) * nbodies), cudaHostAllocDefault);  if (mass == NULL) {fprintf(stderr, "cannot allocate mass\n");  exit(-1);}
  cudaHostAlloc((void**) &posx, (sizeof(float) * nbodies), cudaHostAllocDefault);  if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
  cudaHostAlloc((void**) &posy, (sizeof(float) * nbodies), cudaHostAllocDefault);  if (posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
  cudaHostAlloc((void**) &posz, (sizeof(float) * nbodies), cudaHostAllocDefault);  if (posz == NULL) {fprintf(stderr, "cannot allocate posz\n");  exit(-1);}
  cudaHostAlloc((void**) &velx, (sizeof(float) * nbodies), cudaHostAllocDefault);  if (velx == NULL) {fprintf(stderr, "cannot allocate velx\n");  exit(-1);}
  cudaHostAlloc((void**) &vely, (sizeof(float) * nbodies), cudaHostAllocDefault);  if (vely == NULL) {fprintf(stderr, "cannot allocate vely\n");  exit(-1);}
  cudaHostAlloc((void**) &velz, (sizeof(float) * nbodies), cudaHostAllocDefault);  if (velz == NULL) {fprintf(stderr, "cannot allocate velz\n");  exit(-1);}
  
/*
  mass = (float *)malloc(sizeof(float) * nbodies);  if (mass == NULL) {fprintf(stderr, "cannot allocate mass\n");  exit(-1);}
  posx = (float *)malloc(sizeof(float) * nbodies);  if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
  posy = (float *)malloc(sizeof(float) * nbodies);  if (posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
  posz = (float *)malloc(sizeof(float) * nbodies);  if (posz == NULL) {fprintf(stderr, "cannot allocate posz\n");  exit(-1);}
  velx = (float *)malloc(sizeof(float) * nbodies);  if (velx == NULL) {fprintf(stderr, "cannot allocate velx\n");  exit(-1);}
  vely = (float *)malloc(sizeof(float) * nbodies);  if (vely == NULL) {fprintf(stderr, "cannot allocate vely\n");  exit(-1);}
  velz = (float *)malloc(sizeof(float) * nbodies);  if (velz == NULL) {fprintf(stderr, "cannot allocate velz\n");  exit(-1);}
*/
  if (cudaSuccess != cudaMalloc((void **)&massl, sizeof(float) * nbodies)) fprintf(stderr, "could not allocate massd\n");  CudaTest("couldn't allocate massd");
  if (cudaSuccess != cudaMalloc((void **)&posxl, sizeof(float) * nbodies)) fprintf(stderr, "could not allocate posxd\n");  CudaTest("couldn't allocate posxd");
  if (cudaSuccess != cudaMalloc((void **)&posyl, sizeof(float) * nbodies)) fprintf(stderr, "could not allocate posyd\n");  CudaTest("couldn't allocate posyd");
  if (cudaSuccess != cudaMalloc((void **)&poszl, sizeof(float) * nbodies)) fprintf(stderr, "could not allocate poszd\n");  CudaTest("couldn't allocate poszd");
  if (cudaSuccess != cudaMalloc((void **)&velxl, sizeof(float) * nbparts)) fprintf(stderr, "could not allocate velxd\n");  CudaTest("couldn't allocate velxd");
  if (cudaSuccess != cudaMalloc((void **)&velyl, sizeof(float) * nbparts)) fprintf(stderr, "could not allocate velyd\n");  CudaTest("couldn't allocate velyd");
  if (cudaSuccess != cudaMalloc((void **)&velzl, sizeof(float) * nbparts)) fprintf(stderr, "could not allocate velzd\n");  CudaTest("couldn't allocate velzd");
  if (cudaSuccess != cudaMalloc((void **)&accxl, sizeof(float) * nbparts)) fprintf(stderr, "could not allocate accxd\n");  CudaTest("couldn't allocate accxd");
  if (cudaSuccess != cudaMalloc((void **)&accyl, sizeof(float) * nbparts)) fprintf(stderr, "could not allocate accyd\n");  CudaTest("couldn't allocate accyd");
  if (cudaSuccess != cudaMalloc((void **)&acczl, sizeof(float) * nbparts)) fprintf(stderr, "could not allocate acczd\n");  CudaTest("couldn't allocate acczd");

  if (cudaSuccess != cudaMemcpyToSymbol(nbodiesd, &nbodies, sizeof(int))) 
	  fprintf(stderr, "copying of nbodies to device failed\n"); CudaTest("nbody copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(dthfd, &dthf, sizeof(float))) 
	  fprintf(stderr, "copying of dthf to device failed\n");  CudaTest("dthf copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(epssqd, &epssq, sizeof(float))) 
	  fprintf(stderr, "copying of epssq to device failed\n"); CudaTest("epssq copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(massd, &massl, sizeof(void *))) 
	  fprintf(stderr, "copying of massl to device failed\n"); CudaTest("massl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(posxd, &posxl, sizeof(void *))) 
	  fprintf(stderr, "copying of posxl to device failed\n"); CudaTest("posxl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(posyd, &posyl, sizeof(void *))) 
	  fprintf(stderr, "copying of posyl to device failed\n"); CudaTest("posyl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(poszd, &poszl, sizeof(void *))) 
	  fprintf(stderr, "copying of poszl to device failed\n"); CudaTest("poszl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(velxd, &velxl, sizeof(void *))) 
	  fprintf(stderr, "copying of velxl to device failed\n"); CudaTest("velxl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(velyd, &velyl, sizeof(void *))) 
	  fprintf(stderr, "copying of velyl to device failed\n"); CudaTest("velyl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(velzd, &velzl, sizeof(void *))) 
	  fprintf(stderr, "copying of velzl to device failed\n"); CudaTest("velzl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(accxd, &accxl, sizeof(void *))) 
	  fprintf(stderr, "copying of accxl to device failed\n"); CudaTest("accxl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(accyd, &accyl, sizeof(void *))) 
	  fprintf(stderr, "copying of accyl to device failed\n"); CudaTest("accyl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(acczd, &acczl, sizeof(void *))) 
	  fprintf(stderr, "copying of acczl to device failed\n"); CudaTest("acczl copy to device failed");
 
  generateInput(nbodies, mass, posx, posy, posz, velx, vely, velz);
  blocks = (nbparts + THREADS - 1) / THREADS; 
       
  if (cudaSuccess != cudaMemcpy(massl, mass, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) 
    {fprintf(stderr, "copying of mass to device failed\n");  CudaTest("mass copy to device failed");}
  if (cudaSuccess != cudaMemcpy(posxl, posx, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) 
    {fprintf(stderr, "copying of posx to device failed\n");  CudaTest("posx copy to device failed");}
  if (cudaSuccess != cudaMemcpy(posyl, posy, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) 
    {fprintf(stderr, "copying of posy to device failed\n");  CudaTest("posy copy to device failed");}
  if (cudaSuccess != cudaMemcpy(poszl, posz, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) 
    {fprintf(stderr, "copying of posz to device failed\n");  CudaTest("posz copy to device failed");}
  if (cudaSuccess != cudaMemcpy(velxl, velx, sizeof(float) * nbparts, cudaMemcpyHostToDevice)) 
    {fprintf(stderr, "copying of velx to device failed\n");  CudaTest("velx copy to device failed");}
  if (cudaSuccess != cudaMemcpy(velyl, vely, sizeof(float) * nbparts, cudaMemcpyHostToDevice)) 
    {fprintf(stderr, "copying of vely to device failed\n");  CudaTest("vely copy to device failed");}
  if (cudaSuccess != cudaMemcpy(velzl, velz, sizeof(float) * nbparts, cudaMemcpyHostToDevice)) 
    {fprintf(stderr, "copying of velz to device failed\n");  CudaTest("velz copy to device failed");}

  gettimeofday(&starttime, NULL);
  
  for(int i = 0; i < groups; i++) {  
    for (step = 0; step < timesteps; step++) { 
      ForceCalculationKernel<<<blocks, THREADS, 0, streams[i]>>>(nbparts, (i * nbparts), step);
      IntegrationKernel<<<blocks, THREADS, 0, streams[i]>>>(nbparts, (i * nbparts));
   
      if (cudaSuccess != cudaMemcpyAsync(&posx[i * nbparts], &posxl[i * nbparts], sizeof(float) * nbparts, cudaMemcpyDeviceToHost, streams[i]))
         {fprintf(stderr, "copying of posx from device failed\n");} //CudaTest("posx copy from device failed");}
      if (cudaSuccess != cudaMemcpyAsync(&posy[i * nbparts], &posyl[i * nbparts], sizeof(float) * nbparts, cudaMemcpyDeviceToHost, streams[i]))
         {fprintf(stderr, "copying of posy from device failed\n");} //CudaTest("posy copy from device failed");}
      if (cudaSuccess != cudaMemcpyAsync(&posz[i * nbparts], &poszl[i * nbparts], sizeof(float) * nbparts, cudaMemcpyDeviceToHost, streams[i]))
         {fprintf(stderr, "copying of posz from device failed\n");} //CudaTest("posz copy from device failed");}

      cudaDeviceSynchronize();
      //cudaStreamSynchronize(streams[i]);

      if (cudaSuccess != cudaMemcpyAsync(posxl, posx, sizeof(float) * nbodies, cudaMemcpyHostToDevice, streams[i]))
        {fprintf(stderr, "copying of posx from device failed\n");} //CudaTest("posx copy from device failed");}
      if (cudaSuccess != cudaMemcpyAsync(posyl, posy, sizeof(float) * nbodies, cudaMemcpyHostToDevice, streams[i]))
        {fprintf(stderr, "copying of posx from device failed\n");} //CudaTest("posy copy from device failed");}
      if (cudaSuccess != cudaMemcpyAsync(poszl, posz, sizeof(float) * nbodies, cudaMemcpyHostToDevice, streams[i]))
        {fprintf(stderr, "copying of posx from device failed\n");} //CudaTest("posz copy from device failed");}

      cudaDeviceSynchronize();
      //cudaStreamSynchronize(streams[i]);
    }
  }
    
    gettimeofday(&endtime, NULL);
    runtime = endtime.tv_sec + (endtime.tv_usec / 1000000.0) - starttime.tv_sec - (starttime.tv_usec / 1000000.0);
    //total += runtime;
  //}
/*
  if (cudaSuccess != cudaMemcpy(posx, posxl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) 
	  fprintf(stderr, "copying of posx from device failed\n"); // CudaTest("posx copy from device failed");
  if (cudaSuccess != cudaMemcpy(posy, posyl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) 
	  fprintf(stderr, "copying of posy from device failed\n"); // CudaTest("posy copy from device failed");
  if (cudaSuccess != cudaMemcpy(posz, poszl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) 
	  fprintf(stderr, "copying of posz from device failed\n"); // CudaTest("posz copy from device failed");
  if (cudaSuccess != cudaMemcpy(velx, velxl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) 
	  fprintf(stderr, "copying of velx from device failed\n"); // CudaTest("velx copy from device failed");
  if (cudaSuccess != cudaMemcpy(vely, velyl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) 
	  fprintf(stderr, "copying of vely from device failed\n"); // CudaTest("vely copy from device failed");
  if (cudaSuccess != cudaMemcpy(velz, velzl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) 
	  fprintf(stderr, "copying of velz from device failed\n"); // CudaTest("velz copy from device failed");
*/
 
  printf("\nruntime: %.8f s\n", runtime); 

  for(int i = 0; i < groups; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  for (int i = 0; i < 1; i++) printf("%.2e %.2e %.2e\n", posx[i], posy[i], posz[i]);
  cudaFreeHost(mass);  cudaFreeHost(posx);  cudaFreeHost(posy);  cudaFreeHost(posz);  cudaFreeHost(velx);  cudaFreeHost(vely);  cudaFreeHost(velz);
  cudaFree(massl);  cudaFree(posxl);  cudaFree(posyl);  cudaFree(poszl);  cudaFree(velxl);  cudaFree(velyl);  cudaFree(velzl);  cudaFree(accxl);  cudaFree(accyl);  cudaFree(acczl);
  return 0;
}

