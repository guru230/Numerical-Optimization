import numpy as np
import pandas as pd

# use pandas to load real_estate_dataset.csv
data = pd.read_csv('real_estate_dataset.csv')

# get the number of samples and the number of features
n_samples, n_features = data.shape

print(f'Number of samples, features:{n_samples, n_features}')

#get the names of the columns
columns = data.columns

# save the column names to file for accessing later as text file
np.savetxt('columns.txt', columns, fmt='%s')

# use Square_Feet, Garage_Size, Location_Score, Distance_to_Center as features
X = data[['Square_Feet', 'Garage_Size', 'Location_Score', 'Distance_to_Center']].values

# Use Price as the target
y = data['Price'].values

print(f'X shape: {X.shape}\n')
print(f"data type of X: {X.dtype}\n")

#get the number of samples and features in X
n_samples, n_features = X.shape

#Build a linear model to predict price from the four features in X
# make an array of coeds of the size of n_features+1, initialize to 1

coefs = np.ones(n_features+1)

# predict the price for each sample in X
predictions = X @ coefs[1:] + coefs[0]

X = np.hstack((np.ones((n_samples, 1)), X))

# predict the price for each sample in X
predictions_bydefn = X @ coefs

# see if all the entries in predictions_bydefn and predictions are the same
print("Are the predictions same",np.allclose(predictions, predictions_bydefn))

# calculate the error using predictions and y
errors = y - predictions

#calculate the relative error
relative_errors = errors/y

#calculate the mean of square of errors using loop
loss_loop = 0
for i in range(n_samples):
    loss_loop += errors[i]**2
loss_loop = loss_loop/n_samples

# calculate the mean of square of errors using matrix operations
loss_matrix = errors.T @ errors/n_samples

#compare the two methods of calculating mean squared error
is_diff = np.allclose(loss_loop, loss_matrix)
print(f"Are the loss by direct and matrix same ?{is_diff} \n" )

# print the size of errors, and its L2 norm
print(f'Errors size: {errors.shape}')
print(f'Errors L2 norm: {np.linalg.norm(errors)}')
print(f'L2 norm of relative error: {np.linalg.norm(relative_errors)}')

# What is my optimization problem?
# I want to find the coefficients that minimize the mean squared error
# this problem is called as least squares problem


# Aside
# in Nu = f(Re,Pr); Nu = \alpha*Re^m*Pr^n,  we want to find the values of \alpha, m, n that minimize the error

# Objective function: f(coefs) = 1/n_samples * \sum_{i=1}^{n_samples} (y_i - \sum_{j=1}^{n_features+1} X_{ij} coefs_j)^2

# What is a solution?
# A solution is a set of coefficients such that the objective function is minimized

# How do i find a solution?
# By searching for the coefficients at which the gradient of the objective function is zero 
# Or I can set the gradient of the objective function to zero and solve for the coefficients

# Write the loss_matrix in terms of the data and coeffs
loss_matrix = ((y - X @ coefs).T @ (y - X @ coefs))/n_samples

# calculate the gradient of the loss with respect to the coefficients
grad_matrix = -2/n_samples * X.T @ (y - X @ coefs)

# we set grade_matrix = 0 and solve for coefs
# X.T @ y = X.T @ X @ coefs
# X.T @ X @ coefs = X.T @ y . This is the normal equation
# coefs = (X.T @X)^{-1} @ X.T @ y

coefs = np.linalg.inv(X.T @ X) @ X.T @ y

# save coefs to a file for viewing 
np.savetxt('coefs.csv', coefs, delimiter=',')

# calculate the predictions using the optimal coefficients
predictions_model = X @ coefs

#calculate the errors using the optimal coefficients
errors_model = y - predictions_model

#print the L2 norm of the errors_model
print(f'L2 norm of errors_model: {np.linalg.norm(errors_model)}')

# print the L2 norm of the relative errors_model
rel_errors_model = errors_model/y
print(f'L2 norm of relative errors_model: {np.linalg.norm(rel_errors_model)}')

# Use all the features in the dataset to build a linear model to predict price
X =data.drop('Price', axis=1).values
y = data['Price'].values

# get the number of samples and features in X
n_samples, n_features = X.shape
print(f'Number of samples, features: {n_samples, n_features}')

# solve the liner model using the normal equation
X = np.hstack((np.ones((n_samples, 1)), X))
coefs = np.linalg.inv(X.T @ X) @ X.T @ y

# save coefs to a file all_coefs.csv
np.savetxt('all_coefs.csv', coefs, delimiter=',')

# calculate the rank of X.T @ X
rank_XtX = np.linalg.matrix_rank(X.T @ X)
print(f'Rank of X.T @ X: {rank_XtX}')

# solve the normal equation using matrix decompostion 
# QR Factorization
Q, R = np.linalg.qr(X)

print(f'Q shape: {Q.shape}')
print(f'R shape: {R.shape}')

# Write R to the file to see it 
np.savetxt('R.csv', R, delimiter=',')

# R*coeffs = b 

sol = Q.T @ Q
np.savetxt('sol.csv', sol, delimiter=',')

# X = QR
# X.T @ X = R.T @ Q.T @ Q @ R = R.T @ R
# X.T @ y = R.T @ Q.T @ y
# R*coeffs = Q.T @ y

b = Q.T @ y

print(f'b shape: {b.shape}')
print(f'R shape: {R.shape}')

# loop to solve R*coeffs = b using back substitution
coeffs_qr_loop = np.zeros(n_features+1)

for i in range(n_features, -1, -1):
    coeffs_qr_loop[i] = b[i]
    for j in range(i+1, n_features+1):
        coeffs_qr_loop[i] -= R[i, j]*coeffs_qr_loop[j]
    coeffs_qr_loop[i] = coeffs_qr_loop[i]/R[i, i]
    
# save the coeffs_qr_loop to a file named coeffs_qr_loop.csv
np.savetxt('coeffs_qr_loop.csv', coeffs_qr_loop, delimiter=',')

# Solve the normal equation using SVD
# X = U S V^t

# Eigen decompostion of a Square matrix
# A = V D V^-1
# A^-1 = V D^-1 V^T
# X*coeffs = y
# A = X^T X

# Normal equation: X^T X coeffs = X^T y
# Xdagger = (X^T X)^-1 X^T

#U, S, Vt = np.linalg.svd(X, full_matrices=False)

# Find the inverse of X in least squares sense 
# pseudo inverse of X 

# To complete: Calculate the coeffs_svd using the pseudo inverse of X

# formulate SVD to solve the normal equation
# X = U S V^T
# X^T X = V S^2 V^T
# X^T y = V S U^T y
# coeffs_svd = (X^T X)^-1 X^T y
# coeffs_svd = (V S^(-2) V^T)(V S U^T y) = V S^-1 U^T y

# save the coeffs_svd to a file named coeffs_svd.csv

U, S, Vt = np.linalg.svd(X, full_matrices=False)
coeffs_svd = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y
np.savetxt('coeffs_svd.csv', coeffs_svd, delimiter=',')