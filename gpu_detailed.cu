#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 512
#define BATCH_SIZE 32

// CUDA kernel for computing BKP dynamic programming
__global__ void compute_bkp_kernel(
    const int* f_in,
    int* f_out,
    unsigned int* decision_matrix,
    int w_k,
    int p_k,
    int b_k,
    int c_min,
    int capacity,
    int local_k,
    int b_bits,
    int n_d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int c_hat = c_min + idx;
    
    if (c_hat > capacity) return;
    
    int max_profit = f_in[c_hat];
    int best_j = 0;
    
    // Try different numbers of items of type k
    int max_j = min(b_k, c_hat / w_k);
    for (int j = 1; j <= max_j; j++) {
        int remaining_cap = c_hat - j * w_k;
        if (remaining_cap >= 0) {
            int profit = f_in[remaining_cap] + j * p_k;
            if (profit > max_profit) {
                max_profit = profit;
                best_j = j;
            }
        }
    }
    
    f_out[c_hat] = max_profit;
    
    // Store decision value in compressed format
    int position_in_batch = local_k % n_d;
    int row_idx = local_k / n_d;
    int shift = position_in_batch * b_bits;
    
    unsigned int mask = ((1u << b_bits) - 1);
    unsigned int value = (best_j & mask) << shift;
    
    atomicOr(&decision_matrix[row_idx * (capacity + 1) + c_hat], value);
}

// Calculate minimum capacity for item k
int calculate_c_min(int* weights, int n, int k, int capacity) {
    int sum = 0;
    for (int i = k + 1; i < n; i++) {
        sum += weights[i];
    }
    int c_min = capacity - sum;
    return (c_min > 0) ? c_min : 0;
}

// Reconstruct solution from compressed decision matrix
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
        int batch_idx = k / BATCH_SIZE;
        int local_k = k % BATCH_SIZE;
        
        int row_idx = batch_idx * b_bits + local_k / n_d;
        int position = local_k % n_d;
        int shift = position * b_bits;
        
        unsigned int value = decision_matrix[row_idx * (capacity + 1) + remaining_capacity];
        int num_items = (value >> shift) & mask;
        
        solution[k] = num_items;
        remaining_capacity -= num_items * weights[k];
    }
}

int main(int argc, char** argv) {
    // Input file
    const char* input_file = (argc > 1) ? argv[1] : "input.txt";
    
    // Read input
    FILE* fp = fopen(input_file, "r");
    if (!fp) {
        printf("Error: Cannot open file %s\n", input_file);
        return 1;
    }
    
    int n, capacity;
    fscanf(fp, "%d %d", &n, &capacity);
    
    int* weights = (int*)malloc(n * sizeof(int));
    int* profits = (int*)malloc(n * sizeof(int));
    int* bounds = (int*)malloc(n * sizeof(int));
    
    int b_M = 0;
    for (int i = 0; i < n; i++) {
        fscanf(fp, "%d %d %d", &weights[i], &profits[i], &bounds[i]);
        if (bounds[i] > b_M) b_M = bounds[i];
    }
    fclose(fp);
    
    // Calculate bits needed to store decision values
    int b_bits = 1;
    int temp = b_M + 1;
    while (temp > 2) {
        temp /= 2;
        b_bits++;
    }
    // Round up to power of 2
    int power = 1;
    while (power < b_bits) power *= 2;
    b_bits = power;
    if (b_bits > 32) b_bits = 32;
    
    int n_d = 32 / b_bits;
    
    printf("Problem: n=%d, capacity=%d, max_bound=%d\n", n, capacity, b_M);
    printf("Memory optimization: %d bits per decision, %d decisions per location\n", b_bits, n_d);
    
    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Allocate GPU memory
    int* d_weights;
    int* d_profits;
    int* d_bounds;
    int* d_f0;
    int* d_f1;
    unsigned int* d_decision_batch;
    
    cudaMalloc(&d_weights, n * sizeof(int));
    cudaMalloc(&d_profits, n * sizeof(int));
    cudaMalloc(&d_bounds, n * sizeof(int));
    cudaMalloc(&d_f0, (capacity + 1) * sizeof(int));
    cudaMalloc(&d_f1, (capacity + 1) * sizeof(int));
    cudaMalloc(&d_decision_batch, b_bits * (capacity + 1) * sizeof(unsigned int));
    
    // Copy data to GPU
    cudaMemcpy(d_weights, weights, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_profits, profits, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bounds, bounds, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_f0, 0, (capacity + 1) * sizeof(int));
    cudaMemset(d_f1, 0, (capacity + 1) * sizeof(int));
    
    // Decision matrix on CPU
    int decision_rows = ((n + BATCH_SIZE - 1) / BATCH_SIZE) * b_bits;
    unsigned int* decision_matrix = (unsigned int*)calloc(decision_rows * (capacity + 1), sizeof(unsigned int));
    
    // Process batches
    int num_batches = (n + BATCH_SIZE - 1) / BATCH_SIZE;
    
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        int batch_start = batch_idx * BATCH_SIZE;
        int batch_end = (batch_start + BATCH_SIZE < n) ? batch_start + BATCH_SIZE : n;
        int batch_actual_size = batch_end - batch_start;
        
        // Reset GPU decision matrix for this batch
        cudaMemset(d_decision_batch, 0, b_bits * (capacity + 1) * sizeof(unsigned int));
        
        // Process each item in the batch
        for (int local_k = 0; local_k < batch_actual_size; local_k++) {
            int k = batch_start + local_k;
            
            // Calculate c_min
            int c_min = calculate_c_min(weights, n, k, capacity);
            
            // Number of blocks needed
            int num_capacities = capacity - c_min + 1;
            int num_blocks = (num_capacities + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            
            // Determine which f vector to use
            int use_f0 = (k % 2 == 1);
            int* f_in = use_f0 ? d_f0 : d_f1;
            int* f_out = use_f0 ? d_f1 : d_f0;
            
            // Launch kernel
            compute_bkp_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
                f_in,
                f_out,
                d_decision_batch,
                weights[k],
                profits[k],
                bounds[k],
                c_min,
                capacity,
                local_k,
                b_bits,
                n_d
            );
        }
        
        cudaDeviceSynchronize();
        
        // Copy decision batch to CPU
        int row_start = batch_idx * b_bits;
        int row_end = (row_start + b_bits < decision_rows) ? row_start + b_bits : decision_rows;
        int rows_to_copy = row_end - row_start;
        
        cudaMemcpy(
            decision_matrix + row_start * (capacity + 1),
            d_decision_batch,
            rows_to_copy * (capacity + 1) * sizeof(unsigned int),
            cudaMemcpyDeviceToHost
        );
    }
    
    // Get final result
    int* final_f = (n % 2 == 0) ? d_f0 : d_f1;
    int optimal_value;
    cudaMemcpy(&optimal_value, final_f + capacity, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Reconstruct solution
    int* solution = (int*)malloc(n * sizeof(int));
    reconstruct_solution(decision_matrix, weights, solution, n, capacity, b_bits, n_d);
    
    // Output results
    printf("\n========== RESULTS ==========\n");
    printf("Optimal Value: %d\n", optimal_value);
    printf("Execution Time: %.4f seconds\n", milliseconds / 1000.0);
    printf("============================\n");
    
    // Optional: Print solution vector
    printf("\nSolution (number of each item type):\n");
    for (int i = 0; i < n; i++) {
        if (solution[i] > 0) {
            printf("Item %d: %d copies\n", i, solution[i]);
        }
    }
    
    // Verify solution
    int total_weight = 0;
    int total_profit = 0;
    for (int i = 0; i < n; i++) {
        total_weight += solution[i] * weights[i];
        total_profit += solution[i] * profits[i];
    }
    printf("\nVerification:\n");
    printf("Total weight: %d (capacity: %d)\n", total_weight, capacity);
    printf("Total profit: %d\n", total_profit);
    
    // Cleanup
    cudaFree(d_weights);
    cudaFree(d_profits);
    cudaFree(d_bounds);
    cudaFree(d_f0);
    cudaFree(d_f1);
    cudaFree(d_decision_batch);
    
    free(weights);
    free(profits);
    free(bounds);
    free(decision_matrix);
    free(solution);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
