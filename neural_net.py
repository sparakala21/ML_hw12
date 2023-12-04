import numpy as np

def tanh_activation(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

def mse_loss(predicted, target):
    return 0.5 * np.mean((predicted - target)**2)

def mse_derivative(predicted, target):
    return predicted - target

def forward_pass(inputs, weights_input_hidden, biases_hidden, weights_hidden_output, bias_output):
    # Hidden layer
    hidden_input = np.dot(inputs, weights_input_hidden) + biases_hidden
    hidden_output = tanh_activation(hidden_input)
    
    # Output layer
    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = output_input  # For identity activation

    return hidden_output, final_output

def backward_pass(inputs, hidden_output, final_output, target, weights_input_hidden, biases_hidden, weights_hidden_output, bias_output, learning_rate):
    # Compute gradients for the output layer
    output_gradients = mse_derivative(final_output, target)

    # Update output layer weights and bias
    weights_hidden_output -= learning_rate * np.outer(hidden_output, output_gradients)
    bias_output -= learning_rate * output_gradients.sum()

    # Compute gradients for the hidden layer
    hidden_gradients = np.dot(output_gradients, weights_hidden_output.T) * tanh_derivative(hidden_output)

    # Update hidden layer weights and bias
    weights_input_hidden -= learning_rate * np.outer(inputs, hidden_gradients)
    biases_hidden -= learning_rate * hidden_gradients

    return weights_input_hidden, biases_hidden, weights_hidden_output, bias_output

# Example data
inputs = np.array([2, 1])
target_output = -1

# Initialize weights and biases
np.random.seed(42)  # for reproducibility
weights_input_hidden = np.random.rand(2, 2)
biases_hidden = np.random.rand(2)
weights_hidden_output = np.random.rand(2)
bias_output = np.random.rand()

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# Training loop
for epoch in range(epochs):
    # Forward pass
    hidden_output, final_output = forward_pass(inputs, weights_input_hidden, biases_hidden, weights_hidden_output, bias_output)

    # Backward pass
    weights_input_hidden, biases_hidden, weights_hidden_output, bias_output = backward_pass(inputs, hidden_output, final_output, target_output, weights_input_hidden, biases_hidden, weights_hidden_output, bias_output, learning_rate)

    # Print the loss every 100 epochs
    if epoch % 100 == 0:
        loss = mse_loss(final_output, target_output)
        print(f'Epoch {epoch}, Loss: {loss}')

# Print the final weights and biases
print("\nFinal Weights and Biases:")
print("weights_input_hidden:\n", weights_input_hidden)
print("biases_hidden:\n", biases_hidden)
print("weights_hidden_output:\n", weights_hidden_output)
print("bias_output:\n", bias_output)
