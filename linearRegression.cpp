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

void LinearRegression::printCost(const std::vector<std::vector<double>>& X_train, const std::vector<double>& Y_true, int iteration)
{
        double cost = MSE(X_train, Y_true);
        std::cout << "Iteration: " << iteration << ", Cost: " << cost << std::endl;
}

std::vector<double> LinearRegression::dCdw(const std::vector<std::vector<double>>& X_train, const std::vector<double>& Y_train)
{
    //create a vector for each weight (in this case m) and initialize it to 0.0
    std::vector<double> dcdw(m, 0.0);

    //std::vector<double> Y_hat = predict(X_train);

    for (int i = 0; i < n; ++i) //goes through each row
    {
        //double Y_pred = (*w * X_train[i]) + *b;
        //dCdw += (Y_pred - Y_train[i]) * X_train[i];

        //predict for this row first 
        double Y_hat_i = 0.0;
        for (int j = 0; j < m; ++j)
        {
            Y_hat_i += w[j] * X_train[i][j];
        }
        Y_hat_i += b;

        //now we use it down here 
        for (int j = 0; j < m; ++j) //goes through each column
        {
            dcdw[j] += (Y_hat_i - Y_train[i]) * X_train[i][j];
        }

    }

    for (int j = 0; j < m; ++j)
    {
        dcdw[j] /= n;
    }
    return dcdw;
}

double LinearRegression::dCdb(const std::vector<std::vector<double>>& X_train, const std::vector<double>& Y_train)
{
    double dcdb = 0.0;
    for (int i = 0; i < n; ++i)
    {
        //predict only here first 
        double Y_hat_i = 0.0;
        for (int j = 0; j < m; ++j)
        {
            Y_hat_i += w[j] * X_train[i][j];
        }
        Y_hat_i += b;

        dcdb += Y_hat_i - Y_train[i];
    }
    return dcdb / n;
}

void LinearRegression::update(const std::vector<std::vector<double>>& X_train, const std::vector<double>& Y_train)
{
    std::vector<double> dw = dCdw(X_train, Y_train);
    double db = dCdb(X_train, Y_train);

    //*w -= lr * dw;
    //*b -= lr * db;

    //update weights
    for (int j = 0; j < m; ++j)
    {
        w[j] -= lr * dw[j];
    }
    b -= lr * db;
}

//MOST IMPORTANT ONE
void LinearRegression::fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& Y_train, double alpha, int iterations)
{
    lr = alpha;
    numIterations = iterations;
    
    //number of rows (samples)
    n = X_train.size();

    //number of columns (features)
    m = X_train[0].size();

    //initilize weights to 0.0
    w = std::vector<double>(m, 0.0);
    b = 0.0;

    for (int i = 0; i < numIterations; ++i)
    {
        update(X_train, Y_train);
        //printCost(X_train, Y_train, i);

        //to avoid printing to much to become a mess, print after 100 iterations 
        if (i % 100 == 0)
        {
            printCost(X_train, Y_train, i);
        }
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
    std::vector<double>::const_iterator itr;
    std::cout << "Predictions: " << std::endl;
    for (auto itr = predictions.begin(); itr != predictions.end(); ++itr)
    {
        std::cout << *itr << std::endl;
    }

    std::vector<double>::const_iterator itr_w;
    std::cout << "Optimal weights: " << std::endl;
    for (auto itr_w = w.begin(); itr_w != w.end(); ++itr_w)
    {        
        std::cout << *itr_w << std::endl;
    }

    std::cout << "Optimal bias: " << b << std::endl;
    //std::cout << "Optimal weights: " << *w << std::endl << "Optimal bias: " << *b << std::endl;
}
        