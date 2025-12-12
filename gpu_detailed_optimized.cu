#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector> // 使用 vector 方便管理 Host 記憶體

#define THREADS_PER_BLOCK 512
#define BATCH_SIZE 32

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)


// Kernel: Skip-Zero Atomic
__global__ void compute_bkp_kernel_v5(
    const int* f_in,
    int* f_out,
    unsigned int* decision_matrix, // 完整 GPU 矩陣
    int w_k,
    int p_k,
    int b_k,
    int c_min,
    int capacity,
    int global_k, // 改用 global_k 直接定位
    int b_bits,
    int n_d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int c_hat = c_min + idx;
    
    if (c_hat > capacity) return;
    
    int max_profit = f_in[c_hat];
    int best_j = 0;
    
    int max_limit_by_weight = (w_k > 0) ? (c_hat / w_k) : b_k;
    int max_j = min(b_k, max_limit_by_weight);

    for (int j = 1; j <= max_j; j++) {
        int remaining_cap = c_hat - j * w_k;
        int profit = f_in[remaining_cap] + j * p_k;
        if (profit > max_profit) {
            max_profit = profit;
            best_j = j;
        }
    }
    
    f_out[c_hat] = max_profit;
    
    if (best_j != 0) {
        // 計算全域位置
        int row_idx = global_k / n_d;
        int position_in_batch = global_k % n_d;
        int shift = position_in_batch * b_bits;
        
        unsigned int mask = ((1u << b_bits) - 1);
        unsigned int value = (best_j & mask) << shift;
        
        // 使用 size_t 避免索引溢位 (重要!)
        size_t matrix_idx = (size_t)row_idx * (capacity + 1) + c_hat;
        
        atomicOr(&decision_matrix[matrix_idx], value);
    }
}

// Host 端預先計算 c_min 避免 CPU 迴圈延遲
void precompute_c_mins(const int* weights, int n, int capacity, std::vector<int>& c_mins) {
    long long sum = 0;
    c_mins.resize(n);
    // 從後往前算後綴和
    for (int i = n - 1; i >= 0; i--) {
        long long current_c_min = capacity - sum;
        c_mins[i] = (current_c_min > 0) ? (int)current_c_min : 0;
        sum += weights[i]; // 為下一個累加 (即 i-1 的後綴和包含 i)
    }
    // 修正邏輯：上述迴圈計算的是 "包含自己在內的後綴和"? 
    // 原邏輯: sum of weights[k+1...n-1]
    // 重寫以確保準確:
    sum = 0;
    for (int k = n - 1; k >= 0; k--) {
        long long val = capacity - sum;
        c_mins[k] = (val > 0) ? (int)val : 0;
        sum += weights[k];
    }
}

void reconstruct_solution(
    unsigned int* decision_matrix,
    int* weights,
    int* solution,
    int n,
    int capacity,
    int b_bits,
    int n_d
) {
    int remaining_capacity = capacity;
    unsigned int mask = (1u << b_bits) - 1;
    
    for (int k = n - 1; k >= 0; k--) {
        int row_idx = k / n_d;
        int position = k % n_d;
        int shift = position * b_bits;
        
        size_t matrix_idx = (size_t)row_idx * (capacity + 1) + remaining_capacity;
        unsigned int value = decision_matrix[matrix_idx];
        
        int num_items = (value >> shift) & mask;
        
        solution[k] = num_items;
        remaining_capacity -= num_items * weights[k];
    }
}

int main(int argc, char** argv) {
    const char* input_file = (argc > 1) ? argv[1] : "input.txt";
    
    FILE* fp = fopen(input_file, "r");
    if (!fp) { printf("Error opening file\n"); return 1; }
    
    int n, capacity;
    if (fscanf(fp, "%d %d", &n, &capacity) != 2) return 1;
    
    int* weights = (int*)malloc(n * sizeof(int));
    int* profits = (int*)malloc(n * sizeof(int));
    int* bounds = (int*)malloc(n * sizeof(int));
    
    int b_M = 0;
    for (int i = 0; i < n; i++) {
        fscanf(fp, "%d %d %d", &weights[i], &profits[i], &bounds[i]);
        if (bounds[i] > b_M) b_M = bounds[i];
    }
    fclose(fp);
    
    int b_bits = 1; 
    int temp = b_M + 1; while (temp > 2) { temp /= 2; b_bits++; }
    int power = 1; while (power < b_bits) power *= 2; b_bits = power;
    if (b_bits > 32) b_bits = 32;
    
    int n_d = 32 / b_bits;
    int total_rows = (n + n_d - 1) / n_d;

    printf("Problem: n=%d, C=%d. Config: %d bits, %d items/int. Total Rows: %d\n", n, capacity, b_bits, n_d, total_rows);
    
    // 預計算 C_min
    std::vector<int> c_mins;
    precompute_c_mins(weights, n, capacity, c_mins);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    int *d_f0, *d_f1;
    unsigned int* d_decision_matrix;
    
    CUDA_CHECK(cudaMalloc(&d_f0, (size_t)(capacity + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_f1, (size_t)(capacity + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_f0, 0, (size_t)(capacity + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_f1, 0, (size_t)(capacity + 1) * sizeof(int)));

    // --- 優化核心：在 GPU 上配置完整的決策矩陣 (約 2.4GB) ---
    size_t matrix_size_bytes = (size_t)total_rows * (capacity + 1) * sizeof(unsigned int);
    printf("Allocating %.2f MB VRAM for Decision Matrix... ", matrix_size_bytes / (1024.0 * 1024.0));
    fflush(stdout);
    
    // 嘗試配置 VRAM
    cudaError_t alloc_err = cudaMalloc(&d_decision_matrix, matrix_size_bytes);
    if (alloc_err != cudaSuccess) {
        printf("Failed! Error: %s. (GTX 1080 should have 8GB, check usage)\n", cudaGetErrorString(alloc_err));
        return 1;
    }
    CUDA_CHECK(cudaMemset(d_decision_matrix, 0, matrix_size_bytes));
    printf("Success.\n");

    // 主迴圈：不再有 cudaMemcpy！
    for (int k = 0; k < n; k++) {
        int c_min = c_mins[k];
        int num_capacities = capacity - c_min + 1;
        int num_blocks = (num_capacities + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        int use_f0 = (k % 2 == 1);
        int* f_in = use_f0 ? d_f0 : d_f1;
        int* f_out = use_f0 ? d_f1 : d_f0;
        
        compute_bkp_kernel_v5<<<num_blocks, THREADS_PER_BLOCK>>>(
            f_in, f_out, d_decision_matrix,
            weights[k], profits[k], bounds[k],
            c_min, capacity, k, b_bits, n_d
        );
    }
    
    // 等待全部算完
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int* final_f = (n % 2 == 0) ? d_f0 : d_f1;
    int optimal_value;
    CUDA_CHECK(cudaMemcpy(&optimal_value, final_f + capacity, sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // 只在最後進行一次巨大的拷貝
    printf("Copying result back to CPU (Pinned Memory)...\n");
    unsigned int* decision_matrix;
    CUDA_CHECK(cudaMallocHost((void**)&decision_matrix, matrix_size_bytes));
    CUDA_CHECK(cudaMemcpy(decision_matrix, d_decision_matrix, matrix_size_bytes, cudaMemcpyDeviceToHost));
    
    int* solution = (int*)malloc(n * sizeof(int));
    reconstruct_solution(decision_matrix, weights, solution, n, capacity, b_bits, n_d);
    
    printf("\n========== RESULTS ==========\n");
    printf("Optimal Value: %d\n", optimal_value);
    printf("Execution Time: %.4f seconds\n", milliseconds / 1000.0);
    printf("============================\n");
    
    // Cleanup
    cudaFree(d_f0); cudaFree(d_f1); cudaFree(d_decision_matrix);
    cudaFreeHost(decision_matrix);
    free(weights); free(profits); free(bounds); free(solution);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}