#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>

// 定義 long long 為 ll
typedef long long ll;

// --- BKP 核心計算函數 (修正為 long long) ---
ll boundedKnapsack(int W, const std::vector<int>& w, const std::vector<int>& v, const std::vector<int>& c) {
    int n = w.size();
    
    // 使用 long long 防止溢位
    std::vector<ll> dp_prev(W + 1, 0);
    std::vector<ll> dp_curr(W + 1, 0);
    
    for (int i = 0; i < n; i++) {
        // 複製上一輪狀態
        dp_curr = dp_prev; 
        
        int current_weight = w[i];
        int current_value = v[i];
        int current_count_limit = c[i];

        // 優化: 只有當物品有重量時才需要嚴格檢查 k * weight <= j
        // 如果 weight 為 0 (極少見但可能)，則只受限於 count
        for (int j = 0; j <= W; j++) {
            
            // 計算這個容量下，最多能放幾個這種物品
            int max_k_by_weight = (current_weight > 0) ? (j / current_weight) : current_count_limit;
            int limit = std::min(current_count_limit, max_k_by_weight);

            for (int k = 1; k <= limit; k++) {
                // 計算過程強制轉型為 long long
                ll val = dp_prev[j - k * current_weight] + (ll)k * current_value;
                if (val > dp_curr[j]) {
                    dp_curr[j] = val;
                }
            }
        }
        dp_prev = dp_curr; 
    }
    return dp_prev[W];
}

void processSingleFileWithTimer(const std::string& inputFileName) {
    // (路徑設定部分保持不變，略...)
    // 假設 inputFileName 就是路徑，或者您自行加上 datasets/
    std::string inputFilePath = inputFileName; 
    
    std::ifstream inputFile(inputFilePath);
    if (!inputFile.is_open()) {
        std::cerr << "錯誤: 無法打開 " << inputFilePath << std::endl;
        return;
    }
    
    std::string line;
    int n = 0;
    int W = 0;
    
    // 讀取 Header
    while (std::getline(inputFile, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        if (ss >> n >> W) break;
    }

    std::vector<int> weights, values, counts;
    
    // 讀取 Items
    while (std::getline(inputFile, line)) {
        if (line.empty()) continue;
        // 處理逗號
        for (char &c : line) if (c == ',') c = ' ';
        
        std::stringstream ss(line);
        int wk, pk, bk;
        // 按照您的邏輯: Weight Profit Bound
        if (ss >> wk >> pk >> bk) {
            weights.push_back(wk);
            values.push_back(pk);
            counts.push_back(bk);
        }
    }
    
    std::cout << "正在計算 (N=" << weights.size() << ", W=" << W << ")..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // 🔥 修正 3: 接收 long long 結果
    ll result = boundedKnapsack(W, weights, values, counts);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    
    std::cout << "✅ 計算完成！" << std::endl;
    std::cout << "   最大價值: " << result << std::endl;
    std::cout << "   執行時間: " << std::fixed << std::setprecision(6) << duration.count() << " 秒" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./cpu_check <filename>" << std::endl;
        return 1;
    }
    processSingleFileWithTimer(argv[1]);
    return 0;
}
