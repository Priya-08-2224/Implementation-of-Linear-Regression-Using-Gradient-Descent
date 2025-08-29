# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess dataset by selecting features and target values, converting to float arrays.
2. Standardize features and target using `StandardScaler`.
3. Train linear regression model using gradient descent to optimize weights (`theta`).
4. Standardize new input data and predict output using learned weights.
5. Inverse-transform the prediction to original scale and display result.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: PRIYADHARSHINI J
RegisterNumber: 212224230210

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data=pd.read_csv("50_Startups.csv")
data.head()

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)

print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
 
*/
```

## Output:
# DATA INFORMATION
<img width="846" height="322" alt="image" src="https://github.com/user-attachments/assets/09fc4778-ed93-4697-9eef-69d26ce43496" />

# VALUE OF X
<img width="531" height="738" alt="image" src="https://github.com/user-attachments/assets/ce1cbf54-5726-44b7-8e75-ace656d66c66" />

# VALUE OF X1_SCALED
<img width="572" height="732" alt="image" src="https://github.com/user-attachments/assets/d43c7fe3-fb19-45b4-b124-af81eca7bc47" />

# PREDICTED VALUE
<img width="533" height="205" alt="image" src="https://github.com/user-attachments/assets/e98a4d2a-c52e-4953-95be-cd8d58fa9ae8" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
