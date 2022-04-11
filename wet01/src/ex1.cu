#include "ex1.h"

__device__ 
void prefix_sum(int arr[], int arr_size) 
{
    // int tid = threadIdx.x; 
    // int increment;
    // // TODO: check if blockdim.x should be arr_size
    // for (int stride = 1 ; stride < blockDim.x ; stride *= 2)
    // {
    //     if (tid >= stride)
    //     {
    //         increment = arr[tid - stride];
    //     }
    //     __syncthreads();
    //     if (tid >= stride)
    //     {
    //         arr[tid] += increment;
    //     }
    //     __syncthreads();
    // }
    for (int i = 1 ; i < arr_size ; i++)
    {
        arr[i] += arr[i-1];
    }
}

__global__
void calc_maps(int* cdf, uchar* maps)
{   
    int tid_x = threadIdx.x;
    
    maps[tid_x] = cdf[tid_x] * (COLOR_COUNT - 1) / (TILE_WIDTH * TILE_WIDTH);
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


// Note: our addition. Check if global is OK
__global__
void calc_histogram(int* hist, uchar* all_in, int tile_tid_x, int tile_tid_y)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int row = tile_tid_x * TILE_WIDTH + tid_x;
    int col = tile_tid_y * TILE_WIDTH + tid_y;

    int color_value = all_in[row * IMG_WIDTH + col];

    atomicAdd(&hist[color_value], 1);

    __syncthreads();
}

__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps) 
{
    int tile_tid_x = threadIdx.x;
    int tile_tid_y = threadIdx.y;
    int* hist = (int*) malloc(COLOR_COUNT * sizeof(uchar));

    // TODO: check this
    dim3 threadHist(TILE_WIDTH, TILE_WIDTH);
    calc_histogram<<<1, threadHist>>>(hist, all_in, tile_tid_x, tile_tid_y);

    prefix_sum(hist, COLOR_COUNT);
    
    calc_maps<<<1, COLOR_COUNT>>>(hist, maps);
    
    // TODO
    free(hist);

    interpolate_device(all_in, all_out, maps);

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
        cudaMemcpy(context->in_img, images_in, IMG_HEIGHT * IMG_WIDTH * sizeof(uchar), cudaMemcpyHostToDevice);

        // add GPU kernel invokation here
        dim3 threadBlock(TILE_COUNT, TILE_COUNT);
        process_image_kernel<<<1, threadBlock>>>(context->in_img, context->out_img, context->maps);

        cudaMemcpy(images_out, context->out_img, IMG_HEIGHT * IMG_WIDTH * sizeof(uchar), cudaMemcpyDeviceToHost);
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
