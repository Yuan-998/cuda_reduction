#include <stdio.h>

int reduce_gold(int *data, int len) {
   int res = 0;
   for (int i = 0; i < len; i++) {
      res += data[i];
   }
   return res;
}

__global__ void reduce1(int *g_idata, int *g_odata) {
   extern __shared__ int sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
   sdata[tid] = g_idata[tid];
   __syncthreads();

   for (unsigned int s = 1; s < blockDim.x; s *= 2) {
      if (tid % (2*s) == 0) {
       sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
   }
   
   if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce2(int *g_idata, int *g_odata) {
   extern __shared__ int sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
   sdata[tid] = g_idata[tid];
   __syncthreads();

   for (unsigned int s = 1; s < blockDim.x; s *= 2) {
      int index = 2 * s * tid;
      if (index < blockDim.x) {
       sdata[index] += sdata[index + s];
      }
      __syncthreads();
   }
   
   if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce3(int *g_idata, int *g_odata) {
   extern __shared__ int sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
   sdata[tid] = g_idata[tid];
   __syncthreads();

   for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
      if (tid < s) {
       sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
   }
   
   if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce4(int *g_idata, int *g_odata) {
   extern __shared__ int sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = threadIdx.x + blockIdx.x*2 * blockDim.x;
   sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
   __syncthreads();

   for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
      if (tid < s) {
       sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
   }
   
   if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__device__ void warpReduce(volatile int *sdata, int tid) {
   sdata[tid] += sdata[tid + 32];
   sdata[tid] += sdata[tid + 16];
   sdata[tid] += sdata[tid + 8];
   sdata[tid] += sdata[tid + 4];
   sdata[tid] += sdata[tid + 2];
   sdata[tid] += sdata[tid + 1];
}

__global__ void reduce5(int *g_idata, int *g_odata) {
   extern __shared__ int sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = threadIdx.x + blockIdx.x*2 * blockDim.x;
   sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
   __syncthreads();

   for (unsigned int s = blockDim.x/2; s > 32; s >>= 1) {
      if (tid < s) {
       sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
   }

   if (tid < 32) warpReduce(sdata, tid);
   
   if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__device__ void warpReduceT(volatile int* sdata, int tid) {
   if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
   if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
   if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
   if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
   if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
   if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}


template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata) {
   extern __shared__ int sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = threadIdx.x + blockIdx.x*2 * blockDim.x;
   sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
   __syncthreads();

   for (unsigned int s = blockDim.x/2; s > 32; s >>= 1) {
      if (tid < s) {
       sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
   }

   if(blockSize >= 1024){
        if(tid < 512){
            sdata[tid] += sdata[tid+512];
        }
        __syncthreads();
    }
    if(blockSize >= 512){
        if(tid < 256){
            sdata[tid] += sdata[tid+256];
        }
        __syncthreads();
    }
    if(blockSize >= 256){
        if(tid < 128){
            sdata[tid] += sdata[tid+128];
        }
        __syncthreads();
    }
    if(blockSize >= 128){
        if(tid < 64){
            sdata[tid] += sdata[tid+64];
        }
        __syncthreads();
    }

   if (tid < 32) warpReduceT<blockSize>(sdata, tid);
   
   if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main() {
   int num_element = 1 << 12;
   int num_block = 8;
   int num_thread = num_element / num_block;

   int *h_idata, *d_idata, *h_odata, *d_odata;

   h_idata = (int *)malloc(sizeof(int) * num_element);
   h_odata = (int *)malloc(sizeof(int) * num_element);

   cudaMalloc((void **)&d_idata, sizeof(int)*num_element);
   cudaMalloc((void **)&d_odata, sizeof(int)*num_element);

   for (unsigned int i = 0; i < num_element; i++) {
      h_idata[i] = i;
   }

   cudaMemcpy(d_idata, h_idata, sizeof(int)*num_element, cudaMemcpyHostToDevice);
   reduce1<<<num_block, num_thread>>>(d_idata, d_odata);
   cudaMemcpy(h_odata, d_odata, sizeof(int)*num_element, cudaMemcpyDeviceToHost);

   int sum_gpu = 0, sum = 0;
   for (unsigned int i = 0; i < num_block; i++) {
      sum_gpu += h_odata[i*num_thread];
   }

   sum = reduce_gold(h_idata, num_element);

   printf("error: %d - %d = %d\n", sum_gpu, sum, sum_gpu-sum);

   free(h_idata);
   free(h_odata);
   cudaFree(d_idata);
   cudaFree(d_odata);

   return 0;
}