import matplotlib.pyplot as plt
import numpy as np

# Read data from file
data = np.loadtxt('decision_boundary_data.txt')

# Extract x1, x2, and predicted_output
x1 = data[:, 0]
x2 = data[:, 1]
predicted_output = np.sign(data[:, 2])

# Plot the decision boundary
plt.scatter(x1, x2, c=predicted_output, cmap='viridis')
plt.title('Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Predicted Output')
plt.savefig("question1partA.png", format="png")
