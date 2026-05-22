#include "linearRegression.h"
#include <iostream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

LinearRegression::LinearRegression()
{
    w = std::vector<double>();
    b = 0.0;
    m = 0;
    n = 0;
    //X = std::vector<double>();
    //Y = std::vector<double>();
    lr = 0.01;
    numIterations = 0;
}

LinearRegression::LinearRegression(double alpha, int maxIterations)
{
    lr = alpha;
    numIterations = maxIterations;
    b = 0; 
    m = 0;
    n = 0; 
}

LinearRegression::~LinearRegression()
{

}

int LinearRegression::calculateNumFeatures(const std::vector<std::vector<double>>& X_train)
{
    //size of the columns of the 2D vector X_train will give us the number of features
    return X_train[0].size();
}

double LinearRegression::MSE(const std::vector<std::vector<double>>& X_train, const std::vector<double>& Y_true)
{
    double cost = 0.0;
    int curr_samples = Y_true.size();
    
    std::vector<double> Y_hat = predict(X_train);

    for (int i = 0; i < curr_samples; ++i)
    {
        cost += pow(Y_hat[i]- Y_true[i], 2);
    }

    return cost / curr_samples;
}

void LinearRegression::printCost(const std::vector<double>& X_train, const std::vector<double>& Y_true, double weight, double bias, int iteration)
{
        double cost = MSE(X_train, Y_true);
        std::cout << "Iteration: " << iteration << ", Cost: " << cost << std::endl;
}

double LinearRegression::dCdw(const std::vector<double>& X_train, const std::vector<double>& Y_train, double weight, double bias)
{
    double dCdw = 0.0;
    int m = calSize(Y_train);
    setWeight(weight);
    setBias(bias);

    for (int i = 0; i < m; ++i)
    {
        //double Y_pred = (*w * X_train[i]) + *b;
        //dCdw += (Y_pred - Y_train[i]) * X_train[i];
    }
    return dCdw / m;
}

double LinearRegression::dCdb(const std::vector<double>& X_train, const std::vector<double>& Y_train, double weight, double bias)
{
    double dCdb = 0.0;
    int m = calSize(Y_train);
    setWeight(weight);
    setBias(bias);

    for (int i = 0; i < m; ++i)
    {
        //double Y_pred = (*w * X_train[i]) + *b;
        //dCdb += Y_pred - Y_train[i];
    }
    return dCdb / m;
}

void LinearRegression::update(const std::vector<double>& X_train, const std::vector<double>& Y_train, double weight, double bias)
{
    double dw = dCdw(X_train, Y_train, weight, bias);
    double db = dCdb(X_train, Y_train, weight, bias);

    //*w -= lr * dw;
    //*b -= lr * db;
}

void LinearRegression::fit(const std::vector<double>& X_train, const std::vector<double>& Y_train, double alpha, int iterations)
{
    lr = alpha;
    numIterations = iterations;
    //m = calculateNumFeatures(X_train);
    for (int i = 0; i < numIterations; ++i)
    {
        //update(X_train, Y_train, *w, *b);
        //printCost(X_train, Y_train, *w, *b, i);
    }
}

std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& X_input)
{
    std::vector<double> predictions;
    for (int i = 0; i < X_input.size(); ++i) //loop through the rows
    {
        double Y_hat = 0.0;

        for (int j = 0; j < m; ++j) //loop through each columns
        {
            Y_hat += w[j] * X_input[i][j];
        }

        Y_hat += b;
        predictions.push_back(Y_hat);
    }

    return predictions;
}

void LinearRegression::printPredictions(const std::vector<double>& predictions)
{
    std::cout << "Predictions: " << std::endl;
    for (const auto& pred : predictions)
    {
        std::cout << pred << std::endl;
    }
    //std::cout << "Optimal weights: " << *w << std::endl << "Optimal bias: " << *b << std::endl;
}
        