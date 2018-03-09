import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rawData = pd.read_csv('Data/ccpp_training.csv')
#rawData = pd.read_csv('Data/ccpp.csv')
#rawData = (rawData - rawData.mean()) / rawData.std() # Normalize data
rawData.head()

# Retrieve data for each column
AT = rawData['AT'].values
V = rawData['V'].values
AP = rawData['AP'].values
RH = rawData['RH'].values
PE = rawData['PE'].values

theta = np.array([0, 0, 0, 0, 0]) # Initialize theta array
magnitudeRows = len(AT) # Get number of rows for input variables
Xo = np.ones(magnitudeRows) # Fill Xo with 1's
X = np.array([Xo, AT, V, AP, RH]).T # Initialize input array X with features
Y = np.array(PE) # Initialize output array Y 

# Control speed of gradient descent (learning rate)
alpha = 0.000001 #0.000001

# Cost of the sum of squared residuals (goal is to minimize cost using gradient descent)
def update_cost(X, Y, theta):
    magnitudeRows = len(Y)
    predictedValue = np.dot(X, theta) # Predicted Value of Y = X dot theta_t  
    error = (predictedValue - Y) # Error = predicted value of y - actual value of y 
    J = (1.0) / (2 * magnitudeRows) * np.dot(error.T, error) # J = cost 
    print("Cost = %.4f" % J)
    return J

# Gradient descent to minimize the cost by finding the minima, produces updated values of theta at each epoch
def gradient_descent(X, Y, theta, alpha, epochs):
    
    # Initialize values 
    magnitudeRows = len(Y) 
    logCost = [0] * epochs

    for epoch in range(epochs):
        
	# Hypothesis = X dot theta_t
        #h = X.dot(theta)
        h = np.dot(X, theta)

        # Gradient = (H_t(X) - loss)Xj where loss = H_t(X) - y
        gradient = np.dot(X.T, (h - Y)) / magnitudeRows

        # Update theta from gradient descent
        theta = theta - alpha * gradient

        # Update cost
        currentCost = update_cost(X, Y, theta)

	# Store current cost in log         
        logCost[epoch] = currentCost
        
    return theta, logCost

# Evaluation metric: Root Mean Squared Error (RMSE)
def RMSE(Y, predicted_Y):
    magnitudeRows = len(Y)
    errorSquaredSummation = sum((Y - predicted_Y) ** 2) # eSS = Summation[(Y - predicted_Y) ^ 2]
    rmseValue = np.sqrt(errorSquaredSummation / magnitudeRows) # rmse = sqrt(eSS / sampleSize)
    return rmseValue

def prediction_testing(thetaNew, intercept, tAT, tV, tAP, tRH):
    predictedValue = intercept + (thetaNew[1] * tAT) + (thetaNew[2] * tV) + (thetaNew[3] * tAP) + (thetaNew[4] * tRH)
    return predictedValue

# Run the algorithm 
thetaNew, logCost = gradient_descent(X, Y, theta, alpha, 10000) # 10,000 epochs

# Coefficient of theta
print("\nTheta coefficients: ", thetaNew)

# Display starting and final cost
print("\nStarting Cost: %.4f" % logCost[0])
print("Final Cost: %.4f" % logCost[-1])

# Display equation of fitted line for CCPP data
print("\nFitted Line = %.4f" % thetaNew[0], " + (%.4f" % thetaNew[1], "AT) + (%.4f" % thetaNew[2], "V) + (%.4f" % thetaNew[3], "AP) + (%.4f" % thetaNew[4], "RH)")

# Display measure of "goodness" 
predicted_Y = X.dot(thetaNew)
print("RMSE: %.4f" % RMSE(Y, predicted_Y))

### TESTING VALUES
# Values: 14.96,41.76,1024.07,73.17,463.26
tAT = 14.96
tV = 41.76
tAP = 1024.07
tRH = 73.17
intercept = thetaNew[0]
print("\nPE: Expected = 463.26, Prediction = %.4f" % prediction_testing(thetaNew, intercept, tAT, tV, tAP, tRH))
###

# Display graph of cost function decrease 
fig = plt.figure('Cost vs Epochs')
plt.plot(logCost)
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Convergence')
plt.show()


