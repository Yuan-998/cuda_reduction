#include <stdio.h>

long reduce_gold(long *data, int len) {
   long res = 0;
   for (long i = 0; i < len; i++) {
      res += data[i];
   }
   return res;
}

__global__ void reduce1(long *g_idata, long *g_odata) {
   extern __shared__ int sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
   sdata[tid] = g_idata[i];
   __syncthreads();

   for (unsigned int s = 1; s < blockDim.x; s *= 2) {
      if (tid % (2*s) == 0) {
       sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
   }
   
   if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce2(long *g_idata, long *g_odata) {
   extern __shared__ int sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
   sdata[tid] = g_idata[i];
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

__global__ void reduce3(long *g_idata, long *g_odata) {
   extern __shared__ int sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
   sdata[tid] = g_idata[i];
   __syncthreads();

   for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
      if (tid < s) {
       sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
   }
   
   if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce4(long *g_idata, long *g_odata) {
   extern __shared__ int sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = threadIdx.x + blockIdx.x * 2 * blockDim.x;
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

__global__ void reduce5(long *g_idata, long *g_odata) {
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
__global__ void reduce6(long *g_idata, long *g_odata) {
   extern __shared__ int sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = threadIdx.x + blockIdx.x * blockDim.x*2;
   sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
   __syncthreads();

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

void kernel6 (long *d_idata, long *d_odata, int num_thread, int num_block) {
   switch (num_thread) {
      case 1024:
         reduce6<1024><<<num_block, num_thread>>>(d_idata, d_odata); break;
      case 512:
         reduce6<512><<<num_block, num_thread>>>(d_idata, d_odata); break;
      case 256:
         reduce6<256><<<num_block, num_thread>>>(d_idata, d_odata); break;
      case 128:
         reduce6<128><<<num_block, num_thread>>>(d_idata, d_odata); break;
      case 64:
         reduce6<64><<<num_block, num_thread>>>(d_idata, d_odata); break;
      case 32:
         reduce6<32><<<num_block, num_thread>>>(d_idata, d_odata); break;
      case 16:
         reduce6<16><<<num_block, num_thread>>>(d_idata, d_odata); break;
      case 8:
         reduce6<8><<<num_block, num_thread>>>(d_idata, d_odata); break;
      case 4:
         reduce6<4><<<num_block, num_thread>>>(d_idata, d_odata); break;
      case 2:
         reduce6<2><<<num_block, num_thread>>>(d_idata, d_odata); break;
      case 1:
         reduce6<1><<<num_block, num_thread>>>(d_idata, d_odata); break;
   }
}

__global__ void reduce7(long *d_idata, long *d_odata, int num_element) {
   extern __shared__ int sdata[];
   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
   unsigned int gridSize = blockDim.x * 2 * gridDim.x;

   sdata[tid] = 0;
   while (i < num_element) {
      sdata[tid] += d_idata[i] + d_idata[i + blockDim.x];
      i += gridSize;
   }
   __syncthreads();

   for (unsigned int s = blockDim.x/2; s > 32; s >>= 1) {
      if (tid < s) {
       sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
   }

   if (tid < 32) warpReduce(sdata, tid);

   if (tid == 0) d_odata[blockIdx.x] = sdata[0];
}

int main() {
   int num_element = 1 << 22;
   int num_thread = 256;
   int num_block = num_element / num_thread;

   long *h_idata, *d_idata, *h_odata, *d_odata;

   h_idata = (long *)malloc(sizeof(long) * num_element);
   h_odata = (long *)malloc(sizeof(long) * num_block);

   cudaMalloc((void **)&d_idata, sizeof(long)*num_element);
   cudaMalloc((void **)&d_odata, sizeof(long)*num_block);

   for (long i = 0; i < num_element; i++) {
      h_idata[i] = i;
   }

   long sum_gpu = 0, sum = 0;

   float milli;
   cudaEvent_t start, end;
   cudaEventCreate(&start);
   cudaEventCreate(&end);

   printf("4M elements for reduction\n\n");

   cudaEventRecord(start);
   sum = reduce_gold(h_idata, num_element);
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&milli, start, end);
   printf("CPU: Elapsed time = %.4f ms\n\n", milli);

   // cudaEventRecord(start);
   cudaMemcpy(d_idata, h_idata, sizeof(long)*num_element, cudaMemcpyHostToDevice);
   // cudaEventRecord(end);
   // cudaEventSynchronize(end);
   // cudaEventElapsedTime(&milli, start, end);
   // printf("Bandwidth: %.2f GB/s\n", sizeof(long)*num_element*1.0 / (milli * 1000000));

   cudaEventRecord(start);
   reduce1<<<num_block, num_thread>>>(d_idata, d_odata);
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&milli, start, end);
   printf("Reduction1: Elapsed time = %.4f ms", milli);
   cudaMemcpy(h_odata, d_odata, sizeof(long)*num_block, cudaMemcpyDeviceToHost);
   for (unsigned int i = 0; i < num_block; i++) {
      sum_gpu += h_odata[i];
   }
   printf(sum_gpu-sum == 0 ? "   correct\n\n" : "   error\n\n");
   sum_gpu = 0;

   cudaEventRecord(start);
   reduce2<<<num_block, num_thread>>>(d_idata, d_odata);
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&milli, start, end);
   printf("Reduction2: Elapsed time = %.4f ms", milli);
   cudaMemcpy(h_odata, d_odata, sizeof(long)*num_block, cudaMemcpyDeviceToHost);
   for (unsigned int i = 0; i < num_block; i++) {
      sum_gpu += h_odata[i];
   }
   printf(sum_gpu-sum == 0 ? "   correct\n\n" : "   error\n\n");
   sum_gpu = 0;

   cudaEventRecord(start);
   reduce3<<<num_block, num_thread>>>(d_idata, d_odata);
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&milli, start, end);
   printf("Reduction3: Elapsed time = %.4f ms", milli);
   cudaMemcpy(h_odata, d_odata, sizeof(long)*num_block, cudaMemcpyDeviceToHost);
   for (unsigned int i = 0; i < num_block; i++) {
      sum_gpu += h_odata[i];
   }
   printf(sum_gpu-sum == 0 ? "   correct\n\n" : "   error\n\n");
   sum_gpu = 0;

   cudaEventRecord(start);
   reduce4<<<num_block, num_thread>>>(d_idata, d_odata);
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&milli, start, end);
   printf("Reduction4: Elapsed time = %.4f ms", milli);
   cudaMemcpy(h_odata, d_odata, sizeof(long)*num_block, cudaMemcpyDeviceToHost);
   for (unsigned int i = 0; i < num_block; i++) {
      sum_gpu += h_odata[i];
   }
   printf(sum_gpu-sum == 0 ? "   correct\n\n" : "   error\n\n");
   sum_gpu = 0;

   cudaEventRecord(start);
   reduce5<<<num_block, num_thread>>>(d_idata, d_odata);
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&milli, start, end);
   printf("Reduction5: Elapsed time = %.4f ms", milli);
   cudaMemcpy(h_odata, d_odata, sizeof(long)*num_block, cudaMemcpyDeviceToHost);
   for (unsigned int i = 0; i < num_block; i++) {
      sum_gpu += h_odata[i];
   }
   printf(sum_gpu-sum == 0 ? "   correct\n\n" : "   error\n\n");
   sum_gpu = 0;

   cudaEventRecord(start);
   kernel6(d_idata, d_odata, num_thread, num_block);
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&milli, start, end);
   printf("Reduction6: Elapsed time = %.4f ms", milli);
   cudaMemcpy(h_odata, d_odata, sizeof(long)*num_block, cudaMemcpyDeviceToHost);
   for (unsigned int i = 0; i < num_block; i++) {
      sum_gpu += h_odata[i];
   }
   printf(sum_gpu-sum == 0 ? "   correct\n\n" : "   error\n\n");
   sum_gpu = 0;

   cudaEventRecord(start);
   reduce7<<<num_block, num_thread>>>(d_idata, d_odata, num_element);
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&milli, start, end);
   printf("Reduction7: Elapsed time = %.4f ms", milli);
   cudaMemcpy(h_odata, d_odata, sizeof(long)*num_block, cudaMemcpyDeviceToHost);
   for (unsigned int i = 0; i < num_block; i++) {
      sum_gpu += h_odata[i];
   }
   printf(sum_gpu-sum == 0 ? "   correct\n\n" : "   error\n\n");
   sum_gpu = 0;

   free(h_idata);
   free(h_odata);
   cudaFree(d_idata);
   cudaFree(d_odata);

   return 0;
}