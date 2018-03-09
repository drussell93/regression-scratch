import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

rawData = pd.read_csv('Data/breast-cancer-wisconsin_binary_training.csv')
#rawData = pd.read_csv('Data/breast-cancer-wisconsin_binary.csv')
rawData.head()

CT = rawData['CT'].values
UCSZ = rawData['UCSZ'].values
UCSH = rawData['UCSH'].values
MA = rawData['MA'].values
SECS = rawData['SECS'].values
BN = rawData['BN'].values
BC = rawData['BC'].values
NN = rawData['NN'].values
M = rawData['M'].values
MALIG = rawData['MALIG'].values

theta = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # Initialize theta array
magnitudeRows = len(CT) # Get number of rows for input variables
Xo = np.ones(magnitudeRows) # Fill Xo with 1's
X = np.array([Xo, CT, UCSZ, UCSH, MA, SECS, BN, BC, NN, M]).T # Initialize input array X with features
Y = np.array(MALIG)

# Control speed of gradient descent (learning rate)
alpha = 0.001 #0.001
epochs = 100000 #100,000

# Cost of the sum of squared residuals (goal is to minimize cost using gradient descent)
def update_cost(X, Y, theta):
    magnitudeRows = len(Y)
    predictedValue = sigmoid(np.dot(X, theta)) # Predicted Value of Y = sigmoid(X dot theta_t)  
    class1_cost = -Y * np.log(predictedValue) # Class = 1
    class0_cost = (1 - Y) * np.log(1 - predictedValue) # Class = 0
    cost = class1_cost - class0_cost
    J = cost.sum() / magnitudeRows
    print("Cost = %.4f" % J)
    return J

# Gradient descent to minimize the cost by finding the minima, produces updated values of theta at each epoch
def gradient_descent(X, Y, theta, alpha, epochs):
    
    # Initialize values 
    magnitudeRows = len(Y) 
    logCost = [0] * epochs

    for epoch in range(epochs):
        
	# Hypothesis = X dot theta_t
        h = np.dot(X, theta)

        # Gradient = (H_t(X) - loss)Xj where loss = sigmoid(h) - y 
        gradient = np.dot(X.T, (sigmoid(h) - Y)) / magnitudeRows

        # Update theta from gradient descent
        theta = theta - alpha * gradient

        # Update cost
        currentCost = update_cost(X, Y, theta)

	# Store current cost in log         
        logCost[epoch] = currentCost
        
    return theta, logCost

def sigmoid(t):
    phi = 1. / (1. + np.exp(-t))
    return phi

# Evaluation metric: Root Mean Squared Error (RMSE)
def RMSE(Y, predicted_Y):
    magnitudeRows = len(Y)
    errorSquaredSummation = sum((Y - predicted_Y) ** 2) # eSS = Summation[(Y - predicted_Y) ^ 2]
    rmseValue = np.sqrt(errorSquaredSummation / magnitudeRows) # rmse = sqrt(eSS / sampleSize)
    return rmseValue

# Prediction testing 
def prediction_testing(thetaNew, tCT, tUCSZ, tUCSH, tMA, tSECS, tBN, tBC, tNN, tM, intercept):
    y = intercept + (thetaNew[1] * tCT) + (thetaNew[2] * tUCSZ) + (thetaNew[3] * tUCSH) + (thetaNew[4] * tMA) + (thetaNew[5] * tSECS) + (thetaNew[6] * tBN) + (thetaNew[7] * tBC) + (thetaNew[8] * tNN) + (thetaNew[9] * tM)

    predictedClass = sigmoid(y)
    return predictedClass

# Run the algorithm 
thetaNew, logCost = gradient_descent(X, Y, theta, alpha, epochs) # 100,000 epochs

# Coefficient of theta
print("\nTheta coefficients: ", thetaNew)

# Display starting and final cost
print("\nStarting Cost: %.4f" % logCost[0])
print("Final Cost: %.4f" % logCost[-1])

# Display equation of fitted line for BCWD
print("\nFitted Line = %.4f" % thetaNew[0], " + (%.4f" % thetaNew[1], "CT) + (%.4f" % thetaNew[2], "UCSZ) + (%.4f" % thetaNew[3], "UCSH) + (%.4f" % thetaNew[4], "MA) + (%.4f" % thetaNew[5], "SECS) + (%.4f" % thetaNew[6], "BN) + (%.4f" % thetaNew[7], "BC) + (%.4f" % thetaNew[8], "NN) + (%.4f" % thetaNew[9], "M)")

# Display measure of "goodness" 
predicted_Y = X.dot(thetaNew)
print("RMSE: %.4f" % RMSE(Y, predicted_Y))


### TESTING VALUES
# Benign (0)
tCT = 5
tUCSZ = 1 
tUCSH = 1
tMA = 1
tSECS = 2
tBN = 1
tBC = 3
tNN = 1
tM = 1
intercept = thetaNew[0]
print("\nBenign: Expected = 0, Prediction = %.4f" % prediction_testing(thetaNew, tCT, tUCSZ, tUCSH, tMA, tSECS, tBN, tBC, tNN, tM, intercept))

# Malignant (1)
tCT = 8
tUCSZ = 10 
tUCSH = 10
tMA = 8
tSECS = 7
tBN = 10
tBC = 9
tNN = 7
tM = 1
intercept = thetaNew[0]
print("Malignant: Expected = 1, Prediction = %.4f" % prediction_testing(thetaNew, tCT, tUCSZ, tUCSH, tMA, tSECS, tBN, tBC, tNN, tM, intercept))
###

# Display graph of cost function decrease 
fig = plt.figure('Cost vs Epochs')
plt.plot(logCost)
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Convergence')
plt.show()
