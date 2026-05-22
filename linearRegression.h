#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

class LinearRegression
{
    private:
        //declaring private variables:
        //weight
        //only be able to predict 1 feature
        //double* w;
        
        std::vector<double> w;

        //bias
        double b;

        //number of features
        int m; 

        //samples
        //std::vector<double> X;
        //std::vector<double> Y;
        //number of samples 
        int n;
        //learning rate
        double lr;

        //number of iterations
        int numIterations;

    public:
        //down here using reference and const because the functions are not allowed to keep the
        //data that will be passed in -> avoid unessary copying of data and also to make sure that the data is not modified by the function
        //constructor
        LinearRegression();

        LinearRegression(double alpha, int maxIterations);

        //destructor
        ~LinearRegression();

        //helper functions

        //calculate size 
        int calSize (const std::vector<double>& x)
        {
            return x.size();
        }

        void setWeight(double weight)
        {
            w.push_back(weight);
        }

        void setBias(double bias)
        {
            b = bias;
        }

        //function to calculate number of features
        int calculateNumFeatures(const std::vector<std::vector<double>>& X_train);

        //function to calculate the mean squared error
        double MSE(const std::vector<std::vector<double>>& X_train, const std::vector<double>& Y_true);

        //helper to print cost @each iteration
        void printCost(const std::vector<double>& X_train, const std::vector<double>& Y_true, double weight, double bias, int iteration);

        //gradient function
        //void gradient(const std::vector<double>& X_train, const std::vector<double>& Y_train, const std::vector<double>& Y_pred);
        //calculate dC/dw
        double dCdw(const std::vector<double>& X_train, const std::vector<double>& Y_train, double weight, double bias);

        double dCdb(const std::vector<double>& X_train, const std::vector<double>& Y_train, double weight, double bias);

        //function to update weights and bias
        void update(const std::vector<double>& X_train, const std::vector<double>& Y_train, double weight, double bias);

        //fit function
        void fit(const std::vector<double>& X_train, const std::vector<double>& Y_train, double alpha, int iterations);

        //predict function
        std::vector<double> predict(const std::vector<std::vector<double>>& X_input);

        void printPredictions(const std::vector<double>& predictions);


};
#endif // LINEARREGRESSION_H