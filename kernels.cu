#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#define MASK_WIDTH 11
#define TILE_WIDTH 16
#define RADIUS (MASK_WIDTH / 2)

__constant__ unsigned char d_mask[MASK_WIDTH * MASK_WIDTH];

__global__ void morphology_dilation(const unsigned char *input, unsigned char *output, int width, int height) {
    __shared__ unsigned char shared_input[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int shared_size = TILE_WIDTH + MASK_WIDTH - 1;
    int block_row_start = blockIdx.y * TILE_WIDTH;
    int block_col_start = blockIdx.x * TILE_WIDTH;

    for (int i = ty; i < shared_size; i += blockDim.y) {
        for (int j = tx; j < shared_size; j += blockDim.x) {
            int img_row = block_row_start + i - RADIUS;
            int img_col = block_col_start + j - RADIUS;

            int row_clamped = img_row;
            int col_clamped = img_col;

            if (img_row < 0) row_clamped = 0;
            if (img_row >= height) row_clamped = height - 1;
            if (img_col < 0) col_clamped = 0;
            if (img_col >= width) col_clamped = width - 1;

            shared_input[i][j] = input[row_clamped * width + col_clamped];
        }
    }


    __syncthreads();

    int row_o = block_row_start + ty;
    int col_o = block_col_start + tx;

    if (ty < TILE_WIDTH && tx < TILE_WIDTH && row_o < height && col_o < width) {
        unsigned char max_val = 0;
        for (int i = 0; i < MASK_WIDTH; i++) {
            for (int j = 0; j < MASK_WIDTH; j++) {
                if (d_mask[i * MASK_WIDTH + j]) {
                    unsigned char val = shared_input[ty + i][tx + j];
                    if (val > max_val) max_val = val;
                }
            }
        }
        output[row_o * width + col_o] = max_val;
    }
}

__global__ void morphology_erosion(const unsigned char *input, unsigned char *output, int width, int height) {
    __shared__ unsigned char shared_input[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int shared_size = TILE_WIDTH + MASK_WIDTH - 1;
    int block_row_start = blockIdx.y * TILE_WIDTH;
    int block_col_start = blockIdx.x * TILE_WIDTH;

    for (int i = ty; i < shared_size; i += blockDim.y) {
        for (int j = tx; j < shared_size; j += blockDim.x) {
            int img_row = block_row_start + i - RADIUS;
            int img_col = block_col_start + j - RADIUS;

            int row_clamped = img_row;
            int col_clamped = img_col;

            if (img_row < 0) row_clamped = 0;
            if (img_row >= height) row_clamped = height - 1;
            if (img_col < 0) col_clamped = 0;
            if (img_col >= width) col_clamped = width - 1;

            shared_input[i][j] = input[row_clamped * width + col_clamped];
        }
    }


    __syncthreads();

    int row_o = block_row_start + ty;
    int col_o = block_col_start + tx;

    if (ty < TILE_WIDTH && tx < TILE_WIDTH && row_o < height && col_o < width) {
        unsigned char min_val = 255;
        for (int i = 0; i < MASK_WIDTH; i++) {
            for (int j = 0; j < MASK_WIDTH; j++) {
                if (d_mask[i * MASK_WIDTH + j]) {
                    unsigned char val = shared_input[ty + i][tx + j];
                    if (val < min_val) min_val = val;
                }
            }
        }
        output[row_o * width + col_o] = min_val;
    }
}

__global__ void knn_distance(
    const float *X_train,
    const float *X_test,
    float *distances,
    int train_len,
    int test_len,
    int dim
) {
    int test_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int train_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (test_idx < test_len && train_idx < train_len) {
        float sum = 0.0f;

        for (int i = 0; i < dim; ++i) {
            float diff = X_test[test_idx * dim + i] - X_train[train_idx * dim + i];
            sum += diff * diff;
        }

        distances[test_idx * train_len + train_idx] = sum;  // store squared distance
    }
}
