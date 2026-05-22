import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time 


#update the data similar to the one in the C++ code
X_train = np.array([
    [1.0, 2.0], 
    [2.0, 1.0], 
    [3.0, 4.0], 
    [4.0, 3.0]
])

Y_train = np.array([9.0, 8.0, 19.0, 18.0])

model = LinearRegression()

startTime = time.time()
model.fit(X_train, Y_train)
endTime = time.time()
executionTime = (endTime - startTime) * 1000


print("weight: ", model.coef_)
print("\nbias: ", model.intercept_)

#predicting data 
Y_pred = np.array([[5.0, 6.0]])
prediction = model.predict(Y_pred)
print("\nPrediction: ", prediction)

print(f"\nExecution time: {executionTime:.4f} ms")