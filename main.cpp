#include <iostream>
#include "linearRegression.h"
#include <vector>
#include <algorithm>
#include <math.h>

int main()
{
    //test data 
    std::vector<double> X_train = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> Y_train = {2.0, 4.0, 6.0, 8.0, 10.0};

    LinearRegression modelLR;

    modelLR.fit(X_train, Y_train, 0.01, 1000);

    std::vector<double> testData = {6.0, 7.0, 8.0};
    std::vector<double> predictions = modelLR.predict(testData);
    modelLR.printPredictions(predictions);
    
    return 0;
}