import numpy as np

def calculate_importance_coefficient(att):
    """
    Calculation of the matrix of coefficients of importance

    parameters:
    att (numpy.ndarray): The matrix of normalized values of the attention coefficients of the graphical model, of size n x n x 2

    return:
    numpy.ndarray: Matrix of importance coefficients U, size n x n
    """
    n = att.shape[0]  # Total number of nodes
    U = np.zeros((n, n))  # Initialize the importance coefficient matrix
    total_sum = 0  # The sum is initialized to 0

    # Step 1: Calculate the average attention coefficient for each node pair and accumulate the index values
    for i in range(n):
        for j in range(n):
            s = att[i][j][0] + att[i][j][1]  # Calculate the sum of the two attention coefficients
            avg = s / 2  # Calculation of average values
            total_sum += np.exp(avg)  # cumulative index value
            U[i][j] = avg  # Store the mean value in the U matrix

    # Step 2: Normalize the matrix of importance coefficients
    for i in range(n):
        for j in range(n):
            U[i][j] = np.exp(U[i][j]) / total_sum  # Normalize each coefficient

    return U  # Return the matrix of importance coefficients

# demo
att = np.random.rand(5, 5, 2)  # Generate a randomized 5x5x2 matrix of attention coefficients
U = calculate_importance_coefficient(att)  # Call the function to compute the matrix of importance coefficients
print(U) 
