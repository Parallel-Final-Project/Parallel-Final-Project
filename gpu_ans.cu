#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

// 定義物品結構
struct Item {
    int w; // weight
    int p; // profit
    int b; // bound (count)
};

// 錯誤檢查巨集
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

// --------------------------------------------------------------------------
// CUDA Kernel: 根據 Algorithm 1 實作
// f_old: 上一輪的 DP 表 (f_k-1)
// f_new: 這一輪的 DP 表 (f_k)
// capacity: 背包總容量
// w, p, b: 當前物品的重量、價值、數量限制
// c_min: 優化邊界，只更新大於 c_min 的容量
// --------------------------------------------------------------------------
__global__ void knapsack_kernel(const int* f_old, int* f_new, int capacity, int w, int p, int b, int c_min) {
    int c_hat = blockIdx.x * blockDim.x + threadIdx.x;

    // 論文優化: 只計算 c_min <= c_hat <= capacity 的範圍
    // 以及邊界檢查
    if (c_hat > capacity) return;
    
    if (c_hat < c_min) {
        // 如果小於 c_min，根據論文邏輯，這些狀態對於達到滿載可能沒有幫助，
        // 但為了保持 DP 正確性 (防止讀取到垃圾值)，我們可以繼承舊值或保持原樣。
        // 在標準 DP 中，通常是 f_new[c] = f_old[c]。
        f_new[c_hat] = f_old[c_hat];
        return;
    }

    // 核心遞迴公式 (Equation 2 in paper):
    // f_k(c_hat) = Max { f_k-1(c_hat - j*w) + j*p } 
    // where 0 <= j <= min(b, floor(c_hat/w))
    
    int max_val = -1;
    
    // 計算 j 的上限: min(b, c_hat / w)
    int max_j = c_hat / w;
    if (max_j > b) max_j = b;

    // 尋找最佳的 j
    for (int j = 0; j <= max_j; ++j) {
        int prev_c = c_hat - j * w;
        int current_val = f_old[prev_c] + j * p;
        if (current_val > max_val) {
            max_val = current_val;
        }
    }

    f_new[c_hat] = max_val;
}

// --------------------------------------------------------------------------
// 讀取檔案函式 (依照您的範本)
// --------------------------------------------------------------------------
bool load_problem(const std::string& filename, int& capacity, std::vector<Item>& items) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    int n_items;
    // 假設格式: 第一行是 "物品數量 背包容量"
    if (!(infile >> n_items >> capacity)) {
        std::cerr << "Error: Invalid file format (header)" << std::endl;
        return false;
    }

    items.clear();
    items.reserve(n_items);

    int w, p, b;
    // 讀取每一行: 重量 價值 數量
    while (infile >> w >> p >> b) {
        items.push_back({w, p, b});
    }

    if (items.size() != n_items) {
        std::cout << "Warning: Header said " << n_items << " items, but read " << items.size() << std::endl;
    }
    
    infile.close();
    return true;
}

// --------------------------------------------------------------------------
// 主程式
// --------------------------------------------------------------------------
int main(int argc, char** argv) {
    // 1. Check Arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int capacity = 0;
    std::vector<Item> items;

    // 2. Load Problem
    if (!load_problem(filename, capacity, items)) {
        return 1;
    }

    // 3. Prepare GPU Memory
    // 論文策略: 使用 f0 和 f1 兩個向量交替
    int* d_f0;
    int* d_f1;
    size_t size_bytes = (capacity + 1) * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_f0, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_f1, size_bytes));

    // 初始化 f0 為 0
    CUDA_CHECK(cudaMemset(d_f0, 0, size_bytes));
    // f1 也可以初始化，雖然第一輪會被覆寫
    CUDA_CHECK(cudaMemset(d_f1, 0, size_bytes));

    // 4. Precompute total remaining weight (for c_min optimization)
    // 為了計算 c_min = C - sum(w_i * b_i for i=k+1 to n)
    // 我們先計算所有物品的總重量極限
    long long total_possible_weight = 0;
    std::vector<long long> suffix_weight_sum(items.size() + 1, 0);
    
    for (int i = items.size() - 1; i >= 0; --i) {
        long long item_max_w = (long long)items[i].w * items[i].b;
        suffix_weight_sum[i] = suffix_weight_sum[i+1] + item_max_w;
    }

    // 設定 CUDA Grid/Block
    int threadsPerBlock = 256;
    int blocksPerGrid = (capacity + threadsPerBlock - 1) / threadsPerBlock;

    // 開始計時 (包含資料傳輸與計算，但不包含檔案讀取)
    auto start_time = std::chrono::high_resolution_clock::now();

    // 5. Main Loop (Algorithm 1)
    // k 從 0 到 n-1 (對應論文的 1 to n)
    for (int k = 0; k < items.size(); ++k) {
        int w = items[k].w;
        int p = items[k].p;
        int b = items[k].b;

        // 計算 c_min (Algorithm 1 Line 4)
        // sum_{i=k+1}^{n} w_i 在這裡是 suffix_weight_sum[k+1]
        long long remaining_w = suffix_weight_sum[k+1];
        long long c_min_long = (long long)capacity - remaining_w;
        int c_min = (c_min_long < 0) ? 0 : (int)c_min_long;

        // 決定誰是 old 誰是 new (Ping-Pong)
        // 論文: if k mod 2 == 0 then read f1 write f0... 
        // 為了方便，我們直接用指標交換邏輯：
        // 偶數 k (0, 2...): input d_f0, output d_f1 (注意: 第一次迴圈 k=0, input 應該是初始化的 d_f0)
        // 等等，讓我們根據論文邏輯修正：
        // 初始狀態: f0 = 0, f1 = 0
        // k=1 (index 0): 讀 f0 (全0), 寫 f1
        // k=2 (index 1): 讀 f1, 寫 f0
        
        const int* d_in;
        int* d_out;

        if (k % 2 == 0) {
            d_in = d_f0;
            d_out = d_f1;
        } else {
            d_in = d_f1;
            d_out = d_f0;
        }

        // 啟動 Kernel
        knapsack_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, capacity, w, p, b, c_min);
        
        // 檢查 Kernel 錯誤
        CUDA_CHECK(cudaGetLastError());
    }

    // 等待 GPU 完成
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // 6. 取得結果
    // 根據最後一次寫入的陣列決定結果在哪裡
    // 如果 items.size() 是奇數 (例如 1, 最後一次 k=0 寫入 f1)，結果在 f1
    // 如果 items.size() 是偶數 (例如 2, 最後一次 k=1 寫入 f0)，結果在 f0
    int* d_result = (items.size() % 2 != 0) ? d_f1 : d_f0;
    
    int max_profit = 0;
    // 我們只需要取 d_result[capacity]
    CUDA_CHECK(cudaMemcpy(&max_profit, &d_result[capacity], sizeof(int), cudaMemcpyDeviceToHost));

    // 7. 輸出結果
    std::cout << "Time: " << elapsed.count() << " s" << std::endl;
    std::cout << "Answer: " << max_profit << std::endl;

    // 清理記憶體
    cudaFree(d_f0);
    cudaFree(d_f1);

    return 0;
}
