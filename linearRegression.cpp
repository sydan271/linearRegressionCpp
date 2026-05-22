#include "linearRegression.h"
#include <iostream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;

LinearRegression::LinearRegression()
{
    w = nullptr;
    b = nullptr;
    m = nullptr;
    X = nullptr;
    Y = nullptr;
    lr = nullptr;
    num_iterations = nullptr;
}

LinearRegression::LinearRegression(double alpha, int maxIterations)
{
    lr = new double(alpha);
    num_iterations = new int(maxIterations);
}

LinearRegression::~LinearRegression()
{
    delete w;
    delete b;
    delete m;
    delete lr;
    delete num_iterations;
}
