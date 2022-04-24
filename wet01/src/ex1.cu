#include "ex1.h"


__device__
void get_histogram(int* hist, uchar* all_in, int tile_row, int tile_col)
{
    int tid = threadIdx.x;

    int color_value = 0;
    int index = 0;
    int x_index = (TILE_WIDTH * tile_row) + (tid / THREADS_PER_ROW);
    int y_index = (TILE_WIDTH * tile_col) + ((tid % THREADS_PER_ROW) * THREAD_WORK);

    for(int j = 0 ; j < THREAD_WORK ; j++)
    {
        index = x_index * IMG_WIDTH + y_index + j;
        color_value = all_in[index];
        atomicAdd(&hist[color_value], 1);
    }  
}


__device__
void prefix_sum(int arr[], int arr_size) 
{
    int tid = threadIdx.x; 
    int increment;
    // TODO: check if blockdim.x should be arr_size
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


__device__
void get_maps(int* cdf, uchar* maps, int tile_row, int tile_col)
{
    const int tile_size = TILE_WIDTH*TILE_WIDTH;
    int tid = threadIdx.x;
    int maps_start_index = ((tile_row * TILE_COUNT) + tile_col) * COLOR_COUNT;

    maps[maps_start_index + tid] = (float(cdf[tid]) * (COLOR_COUNT - 1)) / (tile_size);
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
void interpolate_device(uchar* maps , uchar *in_img, uchar* out_img);

__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps) 
{
    __shared__ int hist[COLOR_COUNT * sizeof(int)];

    for (int tile_row = 0 ; tile_row < TILE_COUNT ; tile_row++)
    {
        for (int tile_col = 0 ; tile_col < TILE_COUNT ; tile_col++)
        {
            memset(hist, 0, COLOR_COUNT * sizeof(int));
            __syncthreads();

            get_histogram(hist, all_in, tile_row, tile_col);
            __syncthreads();          
    
            prefix_sum(hist, COLOR_COUNT); 
            __syncthreads();

            get_maps(hist, maps, tile_row, tile_col);
            __syncthreads();
            
        }
    }
    
    interpolate_device(maps, all_in, all_out);
    __syncthreads();    

    return; 
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context 
{
    uchar* in_img;
    uchar* out_img;
    uchar* maps;
    // TODO define task serial memory buffers
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;

    cudaMalloc(&context->in_img, IMG_HEIGHT * IMG_WIDTH * sizeof(uchar));
    cudaMalloc(&context->out_img, IMG_HEIGHT * IMG_WIDTH * sizeof(uchar));
    cudaMalloc(&context->maps, TILE_COUNT * TILE_COUNT * COLOR_COUNT * sizeof(uchar));
    //TODO: allocate GPU memory for a single input image, a single output image, and maps

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out)
{
    for (int i = 0 ; i < N_IMAGES ; i++)
    {
        uchar* cur_images_in = &images_in[i * IMG_WIDTH * IMG_HEIGHT];
        uchar* cur_images_out = &images_out[i * IMG_WIDTH * IMG_HEIGHT];
        cudaMemcpy(context->in_img, cur_images_in, IMG_HEIGHT * IMG_WIDTH * sizeof(uchar), cudaMemcpyHostToDevice);

        process_image_kernel<<<1, THREADS_PER_BLOCK>>>(context->in_img, context->out_img, context->maps);

        cudaMemcpy(cur_images_out, context->out_img, IMG_HEIGHT * IMG_WIDTH * sizeof(uchar), cudaMemcpyDeviceToHost);
    }
    //TODO: in a for loop:
    //   1. copy the relevant image from images_in to the GPU memory you allocated
    //   2. invoke GPU kernel on this image
    //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    //TODO: free resources allocated in task_serial_init
    cudaFree(context->in_img);
    cudaFree(context->out_img);
    cudaFree(context->maps);

    free(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context 
{
    // TODO define bulk-GPU memory buffers
};

/* Allocate GPU memory for all the input images, output images, and maps.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    //TODO: allocate GPU memory for all the input images, output images, and maps

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: copy all input images from images_in to the GPU memory you allocated
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //TODO: copy output images from GPU memory to images_out
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    //TODO: free resources allocated in gpu_bulk_init

    free(context);
}
