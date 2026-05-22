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

        //number of features (columns)
        //w is associated with m because in the formula f = wx + b and so on, w is the one that is multiplied by the number of features (m)
        int m; 

        //samples
        //std::vector<double> X;
        //std::vector<double> Y;
        //number of samples (rows)
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
        
        //function to calculate the mean squared error
        double MSE(const std::vector<std::vector<double>>& X_train, const std::vector<double>& Y_true);

        //helper to print cost @each iteration
        void printCost(const std::vector<std::vector<double>>& X_train, const std::vector<double>& Y_true, int iteration);

        //gradient function
        //void gradient(const std::vector<double>& X_train, const std::vector<double>& Y_train, const std::vector<double>& Y_pred);
        //calculate dC/dw
        std::vector<double> dCdw(const std::vector<std::vector<double>>& X_train, const std::vector<double>& Y_train);

        double dCdb(const std::vector<std::vector<double>>& X_train, const std::vector<double>& Y_train);

        //function to update weights and bias
        void update(const std::vector<std::vector<double>>& X_train, const std::vector<double>& Y_train);

        //fit function
        void fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& Y_train, double alpha, int iterations);

        //predict function
        std::vector<double> predict(const std::vector<std::vector<double>>& X_input);

        void printPredictions(const std::vector<double>& predictions);


};
#endif // LINEARREGRESSION_H