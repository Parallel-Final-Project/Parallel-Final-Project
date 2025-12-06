#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>

// 定義 long long 為 ll，防止價值溢位
typedef long long ll;

// --- BKP 核心計算函數 ---
ll boundedKnapsack(int W, const std::vector<int>& w, const std::vector<int>& v, const std::vector<int>& c) {
    int n = w.size();
    
    // 使用一維陣列滾動優化空間，O(W)
    std::vector<ll> dp_prev(W + 1, 0);
    std::vector<ll> dp_curr(W + 1, 0);
    
    for (int i = 0; i < n; i++) {
        dp_curr = dp_prev; 
        
        int current_weight = w[i];
        int current_value = v[i];
        int current_count_limit = c[i];

        // 遍歷每一個背包容量 j
        for (int j = 0; j <= W; j++) {
            // 計算這個容量下，受限於重量能放幾個
            int max_k_by_weight = (current_weight > 0) ? (j / current_weight) : current_count_limit;
            // 實際能放的數量是：數量限制 與 重量限制 取小者
            int limit = std::min(current_count_limit, max_k_by_weight);

            for (int k = 1; k <= limit; k++) {
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

void processSingleFile(const std::string& inputFilePath) {
    // 1. 設定輸出檔案路徑與名稱
    // 邏輯：在原檔名前面加上 "baseline_"，並保持在原本的資料夾內
    std::string directory = "";
    std::string filename = inputFilePath;
    
    // 分離路徑與檔名
    size_t lastSlash = inputFilePath.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        directory = inputFilePath.substr(0, lastSlash + 1);
        filename = inputFilePath.substr(lastSlash + 1);
    }
    
    // 組合新路徑: datasets/baseline_原檔名.txt
    std::string outputFilePath = directory + "baseline_" + filename;

    std::ifstream inputFile(inputFilePath);
    std::ofstream outputFile(outputFilePath);
    
    if (!inputFile.is_open()) {
        std::cerr << "錯誤: 無法打開輸入檔案 " << inputFilePath << std::endl;
        return;
    }
    if (!outputFile.is_open()) {
        std::cerr << "錯誤: 無法建立輸出檔案 " << outputFilePath << std::endl;
        return;
    }
    
    std::string line;
    int n = 0;
    int W = 0;
    
    // 2. 讀取 Header (n W)
    while (std::getline(inputFile, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        if (ss >> n >> W) break;
    }

    std::vector<int> weights, values, counts;
    
    // 3. 讀取 Items (重量 價值 數量)
    while (std::getline(inputFile, line)) {
        if (line.empty()) continue;
        // 處理逗號，將 ',' 替換為空白，以支援 CSV 格式
        for (char &c : line) if (c == ',') c = ' ';
        
        std::stringstream ss(line);
        int wk, pk, bk;
        // 格式讀取: Weight Profit Bound
        if (ss >> wk >> pk >> bk) {
            weights.push_back(wk);
            values.push_back(pk);
            counts.push_back(bk);
        }
    }
    
    std::cout << "正在計算 " << filename << " (N=" << n << ", W=" << W << ")..." << std::endl;

    // 4. 開始計時與計算
    auto start_time = std::chrono::high_resolution_clock::now();

    ll result = boundedKnapsack(W, weights, values, counts);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    
    // 5. 寫入輸出檔案 (符合您要求的格式)
    // 第一行: n W
    outputFile << n << " " << W << "\n";
    // 第二行: 最大價值
    outputFile << result << "\n";
    // 第三行: 執行時間
    outputFile << std::fixed << std::setprecision(6) << duration.count() << "\n";
    
    // 6. 螢幕顯示結果 (方便確認)
    std::cout << "✅ 計算完成！" << std::endl;
    std::cout << "   最大價值: " << result << std::endl;
    std::cout << "   執行時間: " << std::fixed << std::setprecision(6) << duration.count() << " 秒" << std::endl;
    std::cout << "   結果存至: " << outputFilePath << std::endl;
    std::cout << "--------------------------------------" << std::endl;
}

int main(int argc, char** argv) {
    // 檢查是否有傳入檔案參數
    if (argc < 2) {
        std::cout << "使用方式: ./baseline_final <檔案路徑>" << std::endl;
        std::cout << "範例: ./baseline_final datasets/bkp_n5000_bm100_variable_1.txt" << std::endl;
        return 1;
    }
    
    processSingleFile(argv[1]);
    
    return 0;
}
