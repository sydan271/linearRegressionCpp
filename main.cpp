#include <iostream>
#include "linearRegression.h"
#include <vector>
#include <algorithm>
#include <math.h>
#include <chrono>
#include <fstream>
#include <string>

int main()
{
    // Dataset: y = 2*x1 + 3*x2 + 1
    // X columns: [x1, x2]
    std::vector<std::vector<double>> X_train = {
        {1.0, 2.0}, // 2(1) + 3(2) + 1 = 9
        {2.0, 1.0}, // 2(2) + 3(1) + 1 = 8
        {3.0, 4.0}, // 2(3) + 3(4) + 1 = 19
        {4.0, 3.0}  // 2(4) + 3(3) + 1 = 18
    };

    std::vector<double> Y_train = {9.0, 8.0, 19.0, 18.0};

    // Instantiate model
    LinearRegression model;

    std::cout << "Training..." << std::endl;
    // Train with a small learning rate

    auto start = std::chrono::high_resolution_clock::now();

    model.fit(X_train, Y_train, 0.01, 1000);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Training completed in " << duration.count() << " ms" << std::endl;

    // Predict new data
    // Test: {5.0, 6.0} -> 2(5) + 3(6) + 1 = 29
    std::vector<std::vector<double>> test_data = {{5.0, 6.0}};
    std::vector<double> results = model.predict(test_data);

    model.printPredictions(results);
    return 0;
}