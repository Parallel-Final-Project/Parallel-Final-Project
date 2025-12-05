#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono> // 計時器庫
#include <iomanip> // 輸出小數位數

// --- BKP 核心計算函數 (動態規劃) ---
int boundedKnapsack(int W, const std::vector<int>& w, const std::vector<int>& v, const std::vector<int>& c) {
    int n = w.size();
    // 使用 O(W) 空間複雜度
    std::vector<int> dp_prev(W + 1, 0);
    std::vector<int> dp_curr(W + 1, 0);
    
    for (int i = 1; i <= n; i++) {
        dp_curr = dp_prev; 
        
        int current_weight = w[i - 1];
        int current_value = v[i - 1];
        int current_count_limit = c[i - 1];

        for (int j = 0; j <= W; j++) {
            // 內層迴圈：遍歷選擇當前物品的數量 k
            for (int k = 1; k <= current_count_limit && k * current_weight <= j; k++) {
                dp_curr[j] = std::max(
                    dp_curr[j],
                    dp_prev[j - k * current_weight] + k * current_value
                );
            }
        }
        dp_prev = dp_curr; 
    }
    return dp_prev[W];
}

// --- 檔案讀取、計時、寫入邏輯 ---
void processSingleFileWithTimer(const std::string& inputFileName) {
    const std::string inputFilePath = "datasets/" + inputFileName;
    
    // 輸出檔案名設定
    std::string baseName = inputFileName;
    size_t lastDot = inputFileName.find_last_of('.');
    if (lastDot != std::string::npos) {
        baseName = inputFileName.substr(0, lastDot);
    }
    const std::string outputFilePath = "datasets/" + baseName + "_answer.txt";
    
    std::ifstream inputFile(inputFilePath);
    std::ofstream outputFile(outputFilePath);
    std::stringstream inputContent; 
    
    if (!inputFile.is_open()) {
        std::cerr << "錯誤: 無法打開輸入檔案 " << inputFilePath << "\n請確認 datasets 資料夾內是否有該檔案。" << std::endl;
        return;
    }
    
    std::string line;
    int n = 0;
    int W = 0;
    
    // 1. 讀取第一行：n (物品總類型數) 和 W (背包容量)
    if (std::getline(inputFile, line)) {
        inputContent << line << "\n";
        std::stringstream ss(line);
        if (!(ss >> n >> W)) {
             std::cerr << "格式錯誤: 第一行應為 n W" << std::endl;
             return;
        }

        std::vector<int> weights;
        std::vector<int> values;
        std::vector<int> counts;

        // 2. 讀取後續行：物品數據 (格式: 重量, 價值, 數量)
        while (std::getline(inputFile, line)) {
            if (line.empty()) continue;
            inputContent << line << "\n";
            std::stringstream item_ss(line);
            
            int wk, pk, bk;
            char comma; 

            // 嘗試讀取帶逗號或空格的格式
            // 優先嘗試: 重量 >> 逗號 >> 價值 >> 逗號 >> 數量
            if (item_ss >> wk >> comma >> pk >> comma >> bk) {
                weights.push_back(wk);
                values.push_back(pk);
                counts.push_back(bk);
            } 
            // 備用嘗試: 純空白分隔 (重量 價值 數量)
            else {
                std::stringstream fallback_ss(line);
                if (fallback_ss >> wk >> pk >> bk) {
                    weights.push_back(wk);
                    values.push_back(pk);
                    counts.push_back(bk);
                }
            }
        }
        
        std::cout << "正在計算 " << inputFileName << " (n=" << n << ", W=" << W << ")..." << std::endl;

        // --- 🔥 開始計時 ---
        auto start_time = std::chrono::high_resolution_clock::now();

        // 3. 執行 BKP 演算法
        int result = boundedKnapsack(W, weights, values, counts);

        // --- 🔥 結束計時 ---
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // 計算耗時 (秒)
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;
        double duration = elapsed_seconds.count();

        // 4. 輸出結果到檔案與螢幕
        outputFile << "--- BKP 運算報告 ---\n";
        outputFile << "輸入檔案: " << inputFileName << "\n";
        outputFile << "執行時間: " << std::fixed << std::setprecision(6) << duration << " 秒\n";
        outputFile << "最大可獲得價值: " << result << "\n";
        outputFile << "\n--- 輸入數據內容 (格式: 重量, 價值, 數量) ---\n";
        outputFile << inputContent.str();
        outputFile << "------------------------\n";
        
        std::cout << "✅ 計算完成！" << std::endl;
        std::cout << "   最大價值: " << result << std::endl;
        std::cout << "   執行時間: " << duration << " 秒" << std::endl;
        std::cout << "   結果已寫入: " << outputFilePath << std::endl;

    } else {
        std::cerr << "錯誤: 檔案是空的。" << std::endl;
    }
}

int main() {
    // 指定要讀取的單一檔案名稱
    std::string targetFile = "bkp_n5000_bm100_variable_1.txt";

    processSingleFileWithTimer(targetFile);

    return 0;
}