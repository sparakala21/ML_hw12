#include <iostream>
#include <vector>
#include <list>
#include <cmath>
#include <fstream>

// Hyperbolic tangent function
double tanh_activation(double s) {
    return tanh(s);
}

// Derivative of the hyperbolic tangent function
double tanh_derivative(double s) {
    return 1.0 - std::pow(tanh(s), 2);
}

// Identity function
double identity_activation(double s) {
    return s;
}

// Sign function
double sign_activation(double s) {
    return (s >= 0) ? 1.0 : -1.0;
}

// Forward pass through the neural network
double forward_pass(const std::vector<double>& inputs, const std::vector<std::vector<double>>& weights, const std::vector<double>& biases, const std::vector<double>& output_weights, const std::string& output_activation, int m) {
    // Hidden layer
    std::vector<double> hidden(m);
    for (int i = 0; i < m; ++i) {
        double sum = biases[i];
        for (int j = 0; j < 2; ++j) {
            sum += inputs[j] * weights[j][i];
        }
        hidden[i] = tanh_activation(sum);
    }

    // Output layer
    double output = biases[m];
    for (int i = 0; i < m; ++i) {
        output += hidden[i] * output_weights[i];
    }

    // Apply output activation function
    if (output_activation == "identity") {
        return identity_activation(output);
    } else if (output_activation == "tanh") {
        return tanh_activation(output);
    } else if (output_activation == "sign") {
        return sign_activation(output);
    } else {
        // Default to identity
        return identity_activation(output);
    }
}

double mean_squared_error(const std::vector<double>& predicted, const std::vector<double>& target) {
    double error = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        error += std::pow(predicted[i] - target[i], 2);
    }
    return error / 2.0;
}

// Derivative of the mean squared error loss with respect to predicted values
std::vector<double> mse_derivative(const std::vector<double>& predicted, const std::vector<double>& target) {
    std::vector<double> derivative(predicted.size());
    for (size_t i = 0; i < predicted.size(); ++i) {
        derivative[i] = predicted[i] - target[i];
    }
    return derivative;
}

std::list<double> backward_propagation(std::vector<std::vector<double>>& weights ,std::vector<std::vector<double>>& data, std::vector<double>& classifications )
{
    std::list<double> sensitivities = std::list<double>();
    //
    return sensitivities;
}





int main() {
// Network architecture
    int m = 2;

    // Initialize weights and biases to 0.15
    // (Note: You need to adjust this based on your actual network training)
    std::vector<std::vector<double>> weights(2, std::vector<double>(m, 0.15));
    std::vector<double> biases(m + 1, 0.15);
    std::vector<double> output_weights(m, 0.15);
    std::vector<double> input_data = {2, 1};
    std::vector<std::vector<double>> training_data;
    std::vector<double> hidden;

    training_data.push_back(input_data);
    std::string output_activation = "identity";
    std::vector<std::vector<double>> target_output;
    std::vector<double> classification(1, -1);
    target_output.push_back(classification);




    // Create a file to store data for plotting
    std::ofstream dataFile("decision_boundary_data.txt");

    // Iterate over a range of x values to generate data for the plot
    for (double x1 = -5.0; x1 <= 5.0; x1 += 0.1) {
        for (double x2 = -5.0; x2 <= 5.0; x2 += 0.1) {
            std::vector<double> inputs = {x1, x2};
            double predicted_output = forward_pass(inputs, weights, biases, output_weights, output_activation, m);

            // Write data to file
            dataFile << x1 << " " << x2 << " " << predicted_output << std::endl;
        }
    }

    // Close the file
    dataFile.close();

    return 0;
}
