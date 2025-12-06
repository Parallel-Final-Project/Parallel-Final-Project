#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>
#include <algorithm>

#define THREADS_PER_BLOCK 512
#define BATCH_SIZE 32 // 這裡的 Batch 現在是指 "32 個虛擬物品"

// 虛擬物品結構 (用於二進位拆分)
struct VirtualItem {
    int original_id; // 對應原始物品的 ID
    int w;           // 拆分後的重量
    int p;           // 拆分後的價值
    int count;       // 代表原始物品的數量 (1, 2, 4...)
};

// Modified Kernel: 0/1 Knapsack logic (No inner loop)
__global__ void compute_bkp_binary_kernel(
    const int* f_in,
    int* f_out,
    unsigned int* decision_matrix,
    int w,
    int p,
    int c_min,
    int capacity,
    int local_k,  // 在 Batch 中的索引 (0-31)
    int row_idx   // 在 decision_matrix 中的全域列索引
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int c_hat = c_min + idx;
    
    if (c_hat > capacity) return;
    
    int not_taken = f_in[c_hat];
    int taken = -1;
    
    // 0/1 背包邏輯：只有拿或不拿
    if (c_hat >= w) {
        taken = f_in[c_hat - w] + p;
    }

    // 比較並更新 DP 表
    if (taken > not_taken) {
        f_out[c_hat] = taken;
        
        // 記錄決策: 1 bit (taken)
        // 使用 atomicOr 設定對應的 bit
        // local_k 是 0~31，剛好對應 unsigned int 的 32 個 bits
        atomicOr(&decision_matrix[row_idx * (capacity + 1) + c_hat], (1u << local_k));
    } else {
        f_out[c_hat] = not_taken;
        // 預設是 0 (not taken)，不需要寫入，因為初始化時已歸零
    }
}

// Reconstruct solution mapping virtual items back to original items
void reconstruct_solution(
    unsigned int* decision_matrix,
    const std::vector<VirtualItem>& v_items,
    int* solution, // size should be original n
    int capacity
) {
    int remaining_capacity = capacity;
    int num_v_items = v_items.size();
    
    // 從最後一個虛擬物品往回推
    for (int k = num_v_items - 1; k >= 0; k--) {
        int batch_idx = k / BATCH_SIZE;
        int local_k = k % BATCH_SIZE;
        
        // 二進位拆分後，固定 1 bit 存一個決策，所以一行可以存 32 個虛擬物品
        // row_idx 就是 batch_idx
        int row_idx = batch_idx;
        
        unsigned int pack = decision_matrix[row_idx * (capacity + 1) + remaining_capacity];
        
        // 檢查第 local_k 個 bit 是否為 1
        if ((pack >> local_k) & 1) {
            // 被選中了
            const VirtualItem& item = v_items[k];
            
            // 加回原始物品的計數
            solution[item.original_id] += item.count;
            
            // 扣除重量
            remaining_capacity -= item.w;
        }
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
    
    // 暫存原始數據
    std::vector<int> raw_w(n), raw_p(n), raw_b(n);
    for (int i = 0; i < n; i++) {
        fscanf(fp, "%d %d %d", &raw_w[i], &raw_p[i], &raw_b[i]);
    }
    fclose(fp);
    
    // --- STEP 1: 二進位拆分 (Binary Decomposition) ---
    std::vector<VirtualItem> v_items;
    for (int i = 0; i < n; i++) {
        int count = raw_b[i];
        int k = 1;
        while (count > 0) {
            int take = std::min(k, count);
            v_items.push_back({
                i,              // original_id
                raw_w[i] * take,// decomposed weight
                raw_p[i] * take,// decomposed profit
                take            // decomposed count
            });
            count -= take;
            k <<= 1;
        }
    }
    
    int num_v_items = v_items.size();
    printf("Problem: n=%d (Original) -> %d (Virtual Items), capacity=%d\n", n, num_v_items, capacity);
    
    // --- STEP 2: 預計算後綴重量和 (Suffix Sum) 用於 c_min ---
    // c_min 需要知道 "剩下的虛擬物品總重"
    std::vector<long long> suffix_weight(num_v_items + 1, 0);
    for (int i = num_v_items - 1; i >= 0; i--) {
        suffix_weight[i] = suffix_weight[i+1] + v_items[i].w;
    }

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Allocate GPU memory
    int* d_f0;
    int* d_f1;
    unsigned int* d_decision_batch; // 一個 Batch 只需要存一行 (因為 32 bits = 32 items)
    
    cudaMalloc(&d_f0, (capacity + 1) * sizeof(int));
    cudaMalloc(&d_f1, (capacity + 1) * sizeof(int));
    // 每個 Batch 處理 32 個虛擬物品，剛好填滿一個 unsigned int (32 bits)
    cudaMalloc(&d_decision_batch, 1 * (capacity + 1) * sizeof(unsigned int));
    
    cudaMemset(d_f0, 0, (capacity + 1) * sizeof(int));
    cudaMemset(d_f1, 0, (capacity + 1) * sizeof(int));
    
    // Decision matrix on CPU
    // 總行數 = 總虛擬物品數 / 32
    int decision_rows = (num_v_items + BATCH_SIZE - 1) / BATCH_SIZE;
    unsigned int* decision_matrix = (unsigned int*)calloc(decision_rows * (capacity + 1), sizeof(unsigned int));
    
    printf("Decision Matrix Size: %lu MB\n", (unsigned long)decision_rows * (capacity + 1) * sizeof(unsigned int) / (1024*1024));

    // Process batches of VIRTUAL items
    int num_batches = decision_rows;
    
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        int batch_start = batch_idx * BATCH_SIZE;
        int batch_end = std::min(batch_start + BATCH_SIZE, num_v_items);
        int batch_actual_size = batch_end - batch_start;
        
        // Reset GPU decision batch (歸零)
        cudaMemset(d_decision_batch, 0, (capacity + 1) * sizeof(unsigned int));
        
        for (int local_k = 0; local_k < batch_actual_size; local_k++) {
            int global_k = batch_start + local_k;
            const VirtualItem& item = v_items[global_k];
            
            // Calculate c_min using suffix sum
            // c_min = C - sum(weights of remaining items)
            long long remaining_w = suffix_weight[global_k + 1];
            int c_min = 0;
            if (capacity > remaining_w) {
                c_min = capacity - (int)remaining_w;
            }
            
            // Number of blocks
            int num_capacities = capacity - c_min + 1;
            int num_blocks = (num_capacities + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            
            // Determine f vectors (Ping-Pong)
            int use_f0 = (global_k % 2 == 1);
            int* f_in = use_f0 ? d_f0 : d_f1;
            int* f_out = use_f0 ? d_f1 : d_f0;
            
            // Launch 0/1 Knapsack Kernel
            compute_bkp_binary_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
                f_in,
                f_out,
                d_decision_batch,
                item.w,
                item.p,
                c_min,
                capacity,
                local_k, // 這個值決定了寫入哪個 bit (0-31)
                0        // batch 內只有一行，所以這裡是 0 (相對於 d_decision_batch 的偏移)
            );
        }
        
        cudaDeviceSynchronize();
        
        // Copy decision batch to CPU
        // 這個 Batch 的數據對應到 decision_matrix 的第 batch_idx 行
        cudaMemcpy(
            decision_matrix + batch_idx * (capacity + 1),
            d_decision_batch,
            (capacity + 1) * sizeof(unsigned int),
            cudaMemcpyDeviceToHost
        );
    }
    
    // Get final result
    int* final_f = (num_v_items % 2 == 0) ? d_f0 : d_f1;
    int optimal_value;
    cudaMemcpy(&optimal_value, final_f + capacity, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Reconstruct solution
    int* solution = (int*)calloc(n, sizeof(int)); // Initialize to 0
    reconstruct_solution(decision_matrix, v_items, solution, capacity);
    
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
    long long total_weight = 0;
    long long total_profit = 0;
    for (int i = 0; i < n; i++) {
        total_weight += (long long)solution[i] * raw_w[i];
        total_profit += (long long)solution[i] * raw_p[i];
    }
    printf("\nVerification:\n");
    printf("Total weight: %lld (capacity: %d)\n", total_weight, capacity);
    printf("Total profit: %lld\n", total_profit);
    
    // Cleanup
    cudaFree(d_f0);
    cudaFree(d_f1);
    cudaFree(d_decision_batch);
    
    free(decision_matrix);
    free(solution);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
