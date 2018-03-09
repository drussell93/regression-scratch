import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

normClassData = pd.read_csv('Data/iris_norm.csv')
rawData = pd.read_csv('Data/iris_training.csv')
rawData.head()

# Retrieve data for each column
SW = rawData['SW'].values
SL = rawData['SL'].values
PL = rawData['PL'].values
PW = rawData['PW'].values
CLASS = rawData['CLASS'].values
CLASS_NORM = normClassData['CLASS'].values # Used to calculate RMSE 

theta = np.array([0, 0, 0, 0, 0]) # Initialize theta array
magnitudeRows = len(SW) # Get number of rows for input variables
Xo = np.ones(magnitudeRows) # Fill Xo with 1's
X = np.array([Xo, SL, SW, PL, PW]).T # Initialize input array X with features
Y = np.array(CLASS)
yNorm = np.array(CLASS_NORM)

# Control speed of gradient descent (learning rate)
alpha = 0.001 #0.001
epochs = 10000 #10,000

# Initialize cost log for each flower
logCost = [[0 for x in range(epochs)] for y in range(3)] # 3 list each with length of epochs (10,000) filled with 0s
  
# Cost of the sum of squared residuals (goal is to minimize cost using gradient descent)
def update_cost(X, Y, theta, species):
    magnitudeRows = len(Y)
    predictedValue = sigmoid(np.dot(X, theta)) # Predicted Value of Y = sigmoid(X dot theta_t)  
    class1_cost = -Y * np.log(predictedValue) # Class = 1
    class0_cost = (1 - Y) * np.log(1 - predictedValue) # Class = 0
    cost = class1_cost - class0_cost
    J = cost.sum() / magnitudeRows
    print("Cost[", species, "] = %.4f" % J)
    return J

# Gradient descent to minimize the cost by finding the minima, produces updated values of theta at each epoch
def gradient_descent(X, Y, theta, alpha, epochs, species):
    
    # Initialize values 
    magnitudeRows = len(Y) 

    for epoch in range(epochs):
        
	# Hypothesis = X dot theta_t
        h = np.dot(X, theta)

        # Gradient = (H_t(X) - loss)Xj where loss = sigmoid(h) - Y
        gradient = np.dot(X.T, (sigmoid(h) - Y)) / magnitudeRows

        # Update theta from gradient descent
        theta = theta - alpha * gradient

        # Update cost
        currentCost = update_cost(X, Y, theta, species)

	# Store current cost in log         
        logCost[species][epoch] = currentCost
         
    return theta

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
def prediction_testing(thetaNewFlower, tSL, tSW, tPL, tPW, intercept, flower):
    y = intercept + (thetaNewFlower[flower][1] * tSL) + (thetaNewFlower[flower][2] * tSW) + (thetaNewFlower[flower][3] * tPL) + (thetaNewFlower[flower][4] * tPW)

    predictedClass = sigmoid(y)
    return predictedClass

#One vs Rest
thetaNewFlower = np.zeros((3,5))
flowerName = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
i = 0

for flower in flowerName:
    flowerY = np.array(Y == flower, dtype = int)
    thetaNew = gradient_descent(X, flowerY, theta, alpha, epochs, i)
    thetaNewFlower[i] = thetaNew
    i += 1

# Coefficients of theta for all flowers
print("\nTheta coefficients: ", thetaNewFlower)

# Display starting and final cost for each flower
print("\nStarting Cost Iris-setosa: %.4f" % logCost[0][0])
print("Final Cost Iris-setosa: %.4f" % logCost[0][-1])
print("\nStarting Cost Iris-versicolor: %.4f" % logCost[1][0])
print("Final Cost Iris-versicolor: %.4f" % logCost[1][-1])
print("\nStarting Cost Iris-virginica: %.4f" % logCost[2][0])
print("Final Cost Iris-virginica: %.4f" % logCost[2][-1])


# Display equation of fitted line for iris flower data
print("\nFitted Line Iris-setosa = %.4f" % thetaNewFlower[0][0], " + (%.4f" % thetaNewFlower[0][1], "SL) + (%.4f" % thetaNewFlower[0][2], "SW) + (%.4f" % thetaNewFlower[0][3], "PL) + (%.4f" % thetaNewFlower[0][4], "PW)")
print("\nFitted Line Iris-versicolor = %.4f" % thetaNewFlower[1][0], " + (%.4f" % thetaNewFlower[1][1], "SL) + (%.4f" % thetaNewFlower[1][2], "SW) + (%.4f" % thetaNewFlower[1][3], "PL) + (%.4f" % thetaNewFlower[1][4], "PW)")
print("\nFitted Line Iris-virginica = %.4f" % thetaNewFlower[2][0], " + (%.4f" % thetaNewFlower[2][1], "SL) + (%.4f" % thetaNewFlower[2][2], "SW) + (%.4f" % thetaNewFlower[2][3], "PL) + (%.4f" % thetaNewFlower[2][4], "PW)")

# Display measure of "goodness" for each flower
predicted_Y0 = X.dot(thetaNewFlower[0])
print("\nRMSE Iris-setosa: %.4f" % RMSE(yNorm, predicted_Y0))
predicted_Y1 = X.dot(thetaNewFlower[1])
print("RMSE Iris-versicolor: %.4f" % RMSE(yNorm, predicted_Y1))
predicted_Y2 = X.dot(thetaNewFlower[2])
print("RMSE Iris-virginica: %.4f" % RMSE(yNorm, predicted_Y2))

### TESTING VALUES
# Iris Setosa (1)
#5.1,3.5,1.4,0.2,Iris-setosa
tSL = 5.1
tSW = 3.5
tPL = 1.4
tPW = 0.2
intercept = thetaNewFlower[0][0]
print("\nIris-setosa: Expected = 1, Prediction = %.4f" % prediction_testing(thetaNewFlower, tSL, tSW, tPL, tPW, intercept, 0))

# Iris versicolor (0)
#7,3.2,4.7,1.4,Iris-versicolor
tSL = 5.1
tSW = 2.5
tPL = 3
tPW = 1.1
intercept = thetaNewFlower[1][0]
print("\nIris-versicolor: Expected = 0, Prediction = %.4f" % prediction_testing(thetaNewFlower, tSL, tSW, tPL, tPW, intercept, 1))

# Iris Virginica (1)
# Values: 6.3,3.3,6,2.5,Iris-virginica
tSL = 6.3
tSW = 3.3
tPL = 6
tPW = 2.5
intercept = thetaNewFlower[2][0]
print("\nIris-virginica: Expected = 1, Prediction = %.4f" % prediction_testing(thetaNewFlower, tSL, tSW, tPL, tPW, intercept, 2))
###

# Display graph of cost function decrease for each flower
fig = plt.figure('Cost vs Epochs')
plt.plot(logCost[0])
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Convergence Iris-setosa')
plt.show()

fig = plt.figure('Cost vs Epochs')
plt.plot(logCost[1])
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Convergence Iris-versicolor')
plt.show()

fig = plt.figure('Cost vs Epochs')
plt.plot(logCost[2])
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Convergence Iris-virginica')
plt.show()
