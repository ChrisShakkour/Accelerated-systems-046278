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

// TODO implement a lock
// TODO implement a MPMC queue
// TODO implement the persistent kernel - Done
// TODO implement a function for calculating the threadblocks count - Doing

class queue_server : public image_processing_server
{
private:
    // TODO define queue server context (memory buffers, etc...)

private:
    static uint32_t get_threadblock_count()
    {
        /*
        The GPU supports up to <> threads on all thread blocks. 
        Moreover, the amount of possible shared mem is <>. Each TB uses 2KB shared mem.
        According to nvcc nvlink-options: our implementation uses 29 regs, 40 stack, 160B gmem, 2KB smem, 376B cmem, 0B lmem
        According to nvcc ptxas-options: 29 regs, 32B stack frame, 28Bytes spill stores & loads. 
        We can assume each thread uses 32 register. 
        */
        const uint32_t regs_per_thread = 32;
        const uint32_t threads_per_block = THREADS_COUNT;
        const uint32_t used_smem_per_block = 2048;
        cudaDeviceProp prop; 
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

        const uint32_t smem_per_block = prop.sharedMemPerBlock;
        const uint32_t regs_per_block = prop.regsPerBlock;

        const uint32_t sm_count = prop.multiProcessorCount;
        const uint32_t threads_per_sm = prop.maxThreadsPerMultiProcessor;
        const uint32_t smem_per_sm = prop.sharedMemPerMultiprocessor;
        const uint32_t regs_per_sm = prop.regsPerMultiprocessor;


        const uint32_t thread_block_count_smem_criteria = 
        printf("smem_per_block: %d, regs_per_block: %d\n", smem_per_block, regs_per_block);
        printf("sm_count: %d, threads_per_sm: %d, smem_per_sm: %d, regs_per_sm: %d\n", 
        sm_count,
        threads_per_sm,
        smem_per_sm,
        regs_per_sm);

        return 0;
    }

public:
    queue_server(int threads)
    {
        get_threadblock_count();
        // TODO initialize host state
        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
    }

    ~queue_server() override
    {
        // TODO free resources allocated in constructor
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO push new task into queue if possible
        return false;
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.
        return false;

        // TODO return the img_id of the request that was completed.
        //*img_id = ... 
        return true;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
