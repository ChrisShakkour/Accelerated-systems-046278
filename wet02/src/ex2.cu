#include "ex2.h"
#include <cuda/atomic>

__device__ void prefix_sum(int arr[], int arr_size) {
    const int tid = threadIdx.x; 
    int increment;

    for (int stride = 1 ; stride < blockDim.x ; stride *= 2)
    {
        if (tid >= stride)
        {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride)
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
    __shared__ int hist[COLOR_COUNT * sizeof(int)];
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
    // TODO define stream server context (memory buffers, streams, etc...)

public:
    streams_server()
    {
        // TODO initialize context (memory buffers, streams, etc...)
    }

    ~streams_server() override
    {
        // TODO free resources allocated in constructor
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO place memory transfers and kernel invocation in streams if possible.
        return false;
    }

    bool dequeue(int *img_id) override
    {
        return false;

        // TODO query (don't block) streams for any completed requests.
        //for ()
        //{
            cudaError_t status = cudaStreamQuery(0); // TODO query diffrent stream each iteration
            switch (status) {
            case cudaSuccess:
                // TODO return the img_id of the request that was completed.
                //*img_id = ...
                return true;
            case cudaErrorNotReady:
                return false;
            default:
                CUDA_CHECK(status);
                return false;
            }
        //}
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}

// TODO implement a lock
// TODO implement a MPMC queue
// TODO implement the persistent kernel
// TODO implement a function for calculating the threadblocks count

class queue_server : public image_processing_server
{
private:
    // TODO define queue server context (memory buffers, etc...)
public:
    queue_server(int threads)
    {
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
