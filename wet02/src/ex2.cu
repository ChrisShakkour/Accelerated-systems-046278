#include "ex2.h"
#include <cuda/atomic>

__device__ void prefix_sum(int arr[], int arr_size) {
    const int tid = threadIdx.x; 
    int increment;

    for (int stride = 1 ; stride < arr_size ; stride *= 2)
    {
        if (tid >= stride && tid < arr_size)
        {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < arr_size)
        {
            arr[tid] += increment;
        }
        __syncthreads();
    }
}

/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__
 void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);

__device__
void get_histogram(int* hist, uchar* all_in, int tile_row, int tile_col)
{
    const int tid = threadIdx.x;
    const int thread_work = TILE_WIDTH * TILE_WIDTH / blockDim.x;
    const int threads_per_row = TILE_WIDTH / thread_work;
    const int x_index = (TILE_WIDTH * tile_row) + (tid / threads_per_row);
    const int y_index = (TILE_WIDTH * tile_col) + ((tid % threads_per_row) * thread_work);
    int color_value = 0;
    int index = 0;
    
    for(int j = 0 ; j < thread_work ; j++)
    {
        index = x_index * IMG_WIDTH + y_index + j;
        color_value = all_in[index];
        atomicAdd(&hist[color_value], 1);
    }  
}

__device__
void get_maps(int* cdf, uchar* maps, int tile_row, int tile_col)
{
    const int tid = threadIdx.x;
    if (tid >= COLOR_COUNT)
    {
        return;
    }

    const int tile_size = TILE_WIDTH*TILE_WIDTH;
    const int maps_start_index = ((tile_row * TILE_COUNT) + tile_col) * COLOR_COUNT;

    maps[maps_start_index + tid] = (float(cdf[tid]) * (COLOR_COUNT - 1)) / (tile_size);
}

__device__
void process_image(uchar *in, uchar *out, uchar* maps) 
{
    __shared__ int hist[COLOR_COUNT];
    const int image_offset = IMG_HEIGHT * IMG_WIDTH * blockIdx.x;
    const int maps_offset = COLOR_COUNT * TILE_COUNT * TILE_COUNT * blockIdx.x;

    for (int tile_row = 0 ; tile_row < TILE_COUNT ; tile_row++)
    {
        for (int tile_col = 0 ; tile_col < TILE_COUNT ; tile_col++)
        {
            memset(hist, 0, COLOR_COUNT * sizeof(int));
            __syncthreads();

            get_histogram(hist, in + image_offset, tile_row, tile_col);
            __syncthreads();          
    
            prefix_sum(hist, COLOR_COUNT); 
            __syncthreads();            

            get_maps(hist, maps + maps_offset, tile_row, tile_col);
            __syncthreads();
            
        }
    }
    
    interpolate_device(maps + maps_offset, in + image_offset, out + image_offset);
    __syncthreads();    

    return; 
}

__global__
void process_image_kernel(uchar *in, uchar *out, uchar* maps){
    process_image(in, out, maps);
}

class streams_server : public image_processing_server
{
private:
    static const int available_stream = -1;
    cudaStream_t streams[STREAM_COUNT];
    int stream_to_image[STREAM_COUNT];
    uchar* stream_to_map[STREAM_COUNT];
    uchar* stream_to_imgin[STREAM_COUNT];
    uchar* stream_to_imgout[STREAM_COUNT];

public:
    streams_server()
    {
        for (int i = 0 ; i < STREAM_COUNT ; i++)
        {
            stream_to_image[i] = available_stream;
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            CUDA_CHECK(cudaMalloc(&stream_to_map[i], TILE_COUNT * TILE_COUNT * COLOR_COUNT));
            CUDA_CHECK(cudaMalloc(&stream_to_imgin[i], IMG_WIDTH * IMG_HEIGHT));
            CUDA_CHECK(cudaMalloc(&stream_to_imgout[i], IMG_WIDTH * IMG_HEIGHT));
        }
    }

    ~streams_server() override
    {
        // TODO free resources allocated in constructor
        for (int i = 0 ; i < STREAM_COUNT ; i++)
        {
            CUDA_CHECK(cudaFree(stream_to_map[i]));
            CUDA_CHECK(cudaFree(stream_to_imgin[i]));
            CUDA_CHECK(cudaFree(stream_to_imgout[i]));
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        for (int i = 0 ; i < STREAM_COUNT ; i++)
        {
            if (stream_to_image[i] == available_stream)
            {
                stream_to_image[i] = img_id;
                CUDA_CHECK(cudaMemcpyAsync(stream_to_imgin[i], img_in, IMG_WIDTH * IMG_HEIGHT, cudaMemcpyHostToDevice, streams[i]));
                process_image_kernel<<<1, THREADS_COUNT, 0, streams[i]>>>(img_in, stream_to_imgout[i], stream_to_map[i]);
                CUDA_CHECK(cudaMemcpyAsync(img_out, stream_to_imgout[i], IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToHost, streams[i]));
                return true;
            }
        }

        return false;
    }

    bool dequeue(int *img_id) override
    {
        for (int i = 0 ; i < STREAM_COUNT ; i++)
        {
            if (stream_to_image[i] != available_stream)
            {
            cudaError_t status = cudaStreamQuery(streams[i]); 
            switch (status) {
            case cudaSuccess:
                *img_id = stream_to_image[i];
                stream_to_image[i] = available_stream;
                return true;
            case cudaErrorNotReady:
                return false;
            default:
                CUDA_CHECK(status);
                return false;
            }
            }
        }
        return false;
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}





struct ImageRequest
{
    int img_id;
    uchar *img_in;
    uchar *img_out;
    uchar *img_maps;
};


class TASLock 
{
private:
    cuda::atomic<int> _lock;

public:
__device__ void lock() 
{
    while (_lock.exchange(1, cuda::memory_order_relaxed))
    {
        ;
    }
    cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device);
 }
 
 __device__ void unlock() 
 {
    _lock.store(0, cuda::memory_order_release);
 }
};


template <typename T, uint8_t size> 
class RingBuffer 
{
private:
    static const size_t N = size;
    T _mailbox[N];
    cuda::atomic<size_t> _head{0};
    cuda::atomic<size_t> _tail{0};

public:
    static const int32_t invalid_image = -1;

public:
RingBuffer() = default;

// In order to support non blocking enqueue, push must be non blocking too
__device__ __host__ bool push(const T &data) 
{
   int tail = _tail.load(cuda::memory_order_relaxed);
   if ((tail - _head.load(cuda::memory_order_acquire)) % (2 * N) == N)
   {
        return false;
   }

   // critical section
   _mailbox[_tail % N] = data;
   _tail.store(tail + 1, cuda::memory_order_release);
   
   return true;
}

__device__ __host__ T pop() 
 {
    int head = _head.load(cuda::memory_order_relaxed);
    T item;
    // TODO: check this: head or _head
    if((_tail.load(cuda::memory_order_acquire) - head) % (2 * N) == 0)
    {
        item.img_id = invalid_image;
        return item;
    }
    
    // critical section
    item = _mailbox[_head % N];
    _head.store(head + 1, cuda::memory_order_release);
    return item;
 }
};



// TODO implement a lock - Done
// TODO implement a MPMC queue - Done
// TODO implement the persistent kernel
// TODO implement a function for calculating the threadblocks count - Done

__global__ void persistent_gpu_kernel(RingBuffer<ImageRequest, 16 * 4>* cpu_to_gpu_queue, RingBuffer<ImageRequest, 16 * 4>* gpu_to_cpu_queue)
{
    // TODO - implement a kernel that simply listens to the cpu_to_gpu queue, pops any pending requests, and executes it on a single TB. 
    // After its completion, writes the result back to the gpu_to_cpu queue
    // __shared__ TASLock gpu_lock;
    ImageRequest req;

    while(true)
    {
        //gpu_lock.lock();
        req = cpu_to_gpu_queue->pop(); 
        //gpu_lock.unlock();

        if (req.img_id != RingBuffer<ImageRequest, 16 * 4>::invalid_image)
        {
            process_image(req.img_in, req.img_out, req.img_maps);
            //gpu_lock.lock();
            while(gpu_to_cpu_queue->push(req) == false)
            {
                ;
            }
            //gpu_lock.unlock();
        }
    }
}

class queue_server : public image_processing_server
{
private:
    // TODO define queue server context (memory buffers, etc...)
    uint32_t blocks_count;
    RingBuffer<ImageRequest, 16 * 4>* cpu_to_gpu_queue;
    RingBuffer<ImageRequest, 16 * 4>* gpu_to_cpu_queue;
    // TASLock* gpu_lock;
    // TASLock* cpu_lock;

private:
    static uint32_t get_threadblock_count(int threads_per_block)
    {
        /*
        For our GPU setup (sm-75):
        The GPU supports up to 65536 threads on all thread blocks. 
        Moreover, the amount of possible shared mem is 64KB. Since each TB uses 2KB shared mem, this constraint leaders to 32 TBs. 
        The last constraint is the regs count - total of 65536 regs are available for each SM. 
        According to nvcc nvlink-options: our implementation uses 29 regs (we can assume it rounds up to 32, due makefile), 40 stack, 160B gmem, 2KB smem, 376B cmem, 0B lmem
        According to nvcc ptxas-options: 29 regs, 32B stack frame, 28B spill stores & loads. 
        
        Therefore, the limit for our setup is 65536 registers / 32 (threads count) / threads_per_block
        */
        cudaDeviceProp prop; 
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        struct cudaFuncAttributes func;
		CUDA_CHECK(cudaFuncGetAttributes(&func,process_image_kernel));

        const uint32_t regs_per_thread = 32;                          // upper bound of func.numRegs, as stated by makefile
        const uint32_t used_smem_per_block = func.sharedSizeBytes;    // 2KB - 1KB of our shared mem, 1KB of the given interpolate_device function
        //const uint32_t regs_per_block = prop.regsPerBlock;          // this value must be above threads_per_block (max 1024) * regs_per_thread (32), it is 65536
        //const uint32_t smem_per_block = prop.sharedMemPerBlock;     // this value must be above used_smem_per_block, it is 49152 
        //const uint32_t sm_count = prop.multiProcessorCount;
        const uint32_t threads_per_sm = prop.maxThreadsPerMultiProcessor;
        const uint32_t smem_per_sm = prop.sharedMemPerMultiprocessor;
        const uint32_t regs_per_sm = prop.regsPerMultiprocessor;

        const uint32_t max_threads_count = min(regs_per_sm / regs_per_thread, threads_per_sm);       // hides the thread count constraint
        const uint32_t thread_block_count_regs = max_threads_count / threads_per_block; 
        const uint32_t thread_block_count_smem = smem_per_sm / used_smem_per_block;  
        
        return min(thread_block_count_regs, thread_block_count_smem);
    }

public:
    queue_server(int threads)
    {
        // TODO initialize host state
        blocks_count = get_threadblock_count(threads);    

        //cpu_lock = new TASLock();
        // CUDA_CHECK(cudaMalloc(&gpu_lock, sizeof(TASLock))); - we should allocate and initialize this lock on the GPU side.
        CUDA_CHECK(cudaMallocHost(&cpu_to_gpu_queue, sizeof(RingBuffer<ImageRequest, 16 * 4>)));
        CUDA_CHECK(cudaMallocHost(&gpu_to_cpu_queue, sizeof(RingBuffer<ImageRequest, 16 * 4>)));
        new(cpu_to_gpu_queue) RingBuffer<ImageRequest, 16 * 4>();
        new(gpu_to_cpu_queue) RingBuffer<ImageRequest, 16 * 4>();
        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
        persistent_gpu_kernel<<<blocks_count, threads>>>(cpu_to_gpu_queue, gpu_to_cpu_queue);
    }

    ~queue_server() override
    {
        // TODO free resources allocated in constructor
        cpu_to_gpu_queue->~RingBuffer<ImageRequest, 16 * 4>();
        gpu_to_cpu_queue->~RingBuffer<ImageRequest, 16 * 4>();
        CUDA_CHECK(cudaFreeHost(cpu_to_gpu_queue));
        CUDA_CHECK(cudaFreeHost(gpu_to_cpu_queue));
        //delete(cpu_lock);
        //CUDA_CHECK(cudaFree(gpu_lock));
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO push new task into queue if possible
        ImageRequest req;
        req.img_id = img_id;
        req.img_in = img_in;
        req.img_out = img_out;  // Already allocated as shared host memory by the caller
        CUDA_CHECK(cudaMallocHost(&req.img_maps,  TILE_COUNT * TILE_COUNT * COLOR_COUNT));

        bool result = cpu_to_gpu_queue->push(req);
        if (result == false)
        {
            CUDA_CHECK(cudaFree(req.img_maps));
        }

        return result;
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.
        ImageRequest req = gpu_to_cpu_queue->pop();
        if (req.img_id == RingBuffer<ImageRequest, 16 * 4>::invalid_image)
        {
            // must implement non blocking pop to support this
            return false;
        }

        // TODO return the img_id of the request that was completed.
        *img_id = req.img_id;
        // TODO: check this
        CUDA_CHECK(cudaFreeHost(req.img_maps));

        return true;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
