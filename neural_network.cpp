#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

class NeuralNetwork {
public:
    // Constructor
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize) {
        // Initialize weights and biases
        initializeWeights();
    }

    // Destructor
    ~NeuralNetwork() {}

    // Forward pass
    std::vector<double> predict(const std::vector<double>& input) {
        // Calculate hidden layer output
        std::vector<double> hiddenLayerOutput(hiddenSize);
        for (int i = 0; i < hiddenSize; ++i) {
            hiddenLayerOutput[i] = sign_activation(dotProduct(input, weightsInputHidden[i]) + biasesHidden[i]);
        }

        // Calculate output layer output
        std::vector<double> outputLayerOutput(outputSize);
        for (int i = 0; i < outputSize; ++i) {
            outputLayerOutput[i] = sign_activation(dotProduct(hiddenLayerOutput, weightsHiddenOutput[i]) + biasesOutput[i]);
        }

        return outputLayerOutput;
    }

        void backpropagate(const std::vector<double>& input, const std::vector<double>& target, double learningRate) {
        // Forward pass
        std::vector<double> hiddenLayerOutput(hiddenSize);
        for (int i = 0; i < hiddenSize; ++i) {
            hiddenLayerOutput[i] = tanh_activation(dotProduct(input, weightsInputHidden[i]) + biasesHidden[i]);
        }

        std::vector<double> outputLayerOutput(outputSize);
        for (int i = 0; i < outputSize; ++i) {
            outputLayerOutput[i] = sign_activation(dotProduct(hiddenLayerOutput, weightsHiddenOutput[i]) + biasesOutput[i]);
        }

        // Backward pass
        // Calculate output layer error
        std::vector<double> outputErrors(outputSize);
        for (int i = 0; i < outputSize; ++i) {
            outputErrors[i] = target[i] - outputLayerOutput[i];
        }

        // Calculate hidden layer error
        std::vector<double> hiddenErrors(hiddenSize);
        for (int i = 0; i < hiddenSize; ++i) {
            double weightedErrorSum = dotProduct(outputErrors, getColumn(weightsHiddenOutput, i));
            hiddenErrors[i] = hiddenLayerOutput[i] * (1 - hiddenLayerOutput[i]) * weightedErrorSum;
        }

        // Update weights and biases
        // Update hidden to output layer weights and biases
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < hiddenSize; ++j) {
                weightsHiddenOutput[i][j] += learningRate * outputErrors[i] * hiddenLayerOutput[j];
            }
            biasesOutput[i] += learningRate * outputErrors[i];
        }

        // Update input to hidden layer weights and biases
        for (int i = 0; i < hiddenSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                weightsInputHidden[i][j] += learningRate * hiddenErrors[i] * input[j];
            }
            biasesHidden[i] += learningRate * hiddenErrors[i];
        }
    }

    double calculateInSampleError(const std::vector<std::vector<double>>& inputData, const std::vector<double>& targetData) {
        double totalError = 0.0;

        for (size_t i = 0; i < inputData.size(); ++i) {
            // Make a prediction
            std::vector<double> output = predict(inputData[i]);

            // Calculate the error for each output node
            for (int j = 0; j < outputSize; ++j) {
                totalError += pow(targetData[i] - output[j], 2);
            }
        }

        // Calculate mean squared error
        double meanSquaredError = totalError / (2 * inputData.size());
        return meanSquaredError;
    }


private:
    int inputSize;
    int hiddenSize;
    int outputSize;

    // Weights and biases
    std::vector<std::vector<double>> weightsInputHidden;
    std::vector<std::vector<double>> weightsHiddenOutput;
    std::vector<double> biasesHidden;
    std::vector<double> biasesOutput;

    double sign_activation(double s) 
    {
        return (s >= 0) ? 1.0 : -1.0;
    }
    // Hyperbolic tangent function
    double tanh_activation(double s) 
    {
        return tanh(s);
    }

    // Dot product of two vectors
    double dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
        double result = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    // Initialize weights and biases with random values
    void initializeWeights() {
        // Initialize weights and biases for the hidden layer
        for (int i = 0; i < hiddenSize; ++i) {
            weightsInputHidden.push_back(std::vector<double>(inputSize, getRandomValue()));
            biasesHidden.push_back(0.15);
        }

        // Initialize weights and biases for the output layer
        for (int i = 0; i < outputSize; ++i) {
            weightsHiddenOutput.push_back(std::vector<double>(hiddenSize, getRandomValue()));
            biasesOutput.push_back(0.15);
        }
    }




    // Generate a random value between -1 and 1
    double getRandomValue() {
        return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    private:

    // Helper function to get a column from a 2D vector
    std::vector<double> getColumn(const std::vector<std::vector<double>>& matrix, int col) {
        std::vector<double> column;
        for (size_t i = 0; i < matrix.size(); ++i) {
            column.push_back(matrix[i][col]);
        }
        return column;
    }
};

int main() {
    // Create a neural network with 2 input nodes, 3 hidden nodes, and 1 output node
    NeuralNetwork neuralNetwork(2, 2, 1);

    // Example input
    std::vector<std::vector<double>> inputs;
    std::vector<double> targets;


    std::ifstream file("data.txt");

    if (!file.is_open()) {
        std::cerr << "Error opening the file." << std::endl;
        return 1;
    }

    // Read data from the file
    std::vector<std::vector<double>> data;
    double value;

    while (file >> value) {
        std::vector<double> row;
        row.push_back(value);
        for (int i = 0; i < 2; ++i) {
            file >> value;
            row.push_back(value);
        }
        data.push_back(row);
    }

    // Close the file
    file.close();

    for(int i=0 ; i < data.size() ; ++i)
    {
        std::vector<double> point = {data[i][0], data[i][1]};
        inputs.push_back(point);

        double c = data[i][2];
        targets.push_back(c);
    }

    std::cout << std::endl;
    for (int epoch = 0; epoch < 20000000; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Make a prediction
            std::vector<double> output = neuralNetwork.predict(inputs[i]);

            // Backpropagate to update weights and biases
            neuralNetwork.backpropagate(inputs[i], {targets[i]}, 0.1);
        }
        if((epoch%10000000==0)||(epoch==19999999))
        {
            std::cout<<"Epoch "<<epoch<<": "<<"Error: "<<neuralNetwork.calculateInSampleError(inputs, targets)<<std::endl;
        }
    }
    return 0;
}
