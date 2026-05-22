#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;

class LinearRegression
{
    private:
        //declaring private variables:
        //weight
        double* w; 

        //bias
        double* b;

        //number of features
        int* m; 

        //samples
        vector<double>* X;
        vector<double>* Y;

        //learning rate
        double* lr;

        //number of iterations
        int* num_iterations;

    public:
        //constructor
        LinearRegression();

        LinearRegression(double alpha, int maxIterations);

        //destructor
        ~LinearRegression()
        {
            delete w;
            delete b;
            delete m;
            delete lr;
            delete num_iterations;
        }

        //function to calculate number of features
        int calculateNumFeatures(vector<double>* X_train);

        //function to calculate the mean squared error
        double MSE(vector<double>* Y_true, vector<double>* Y_pred);

        //gradient function
        void gradient(vector<double>* X_train, vector<double>* Y_train, vector<double>* Y_pred);

        //function to update weights and bias
        void update(vector<double>* X_train, vector<double>* Y_train, vector<double>* Y_pred);

        //fit function
        void fit(vector<double>* X_train, vector<double>* Y_train, int num_iterations);

        //predict function
        vector<double> predict(vector<double>* X_test);



};
#endif // LINEARREGRESSION_H