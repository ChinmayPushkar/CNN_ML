#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace std;

// Structure to hold image data
struct Image {
    vector<vector<double>> data;
    int label;
};

vector<Image> readMNISTImages(const string& imageFile, const string& labelFile, int numImages) {
    vector<Image> images;
    
    ifstream imagesFile(imageFile, ios::binary);
    ifstream labelsFile(labelFile, ios::binary);
    
    if (!imagesFile.is_open() || !labelsFile.is_open()) {
        cout << "Error opening MNIST files" << endl;
        return images;
    }

    // Skip headers
    imagesFile.seekg(16);  // Magic number (4 bytes), num images (4), rows (4), cols (4)
    labelsFile.seekg(8);   // Magic number (4 bytes), num items (4)

    for (int i = 0; i < numImages; i++) {
        Image img;
        img.data.resize(28, vector<double>(28));
        
        // Read image data
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                unsigned char pixel;
                imagesFile.read((char*)&pixel, 1);
                img.data[r][c] = pixel / 255.0;  // Normalize to [0,1]
            }
        }
        
        // Read label
        unsigned char label;
        labelsFile.read((char*)&label, 1);
        img.label = label;
        
        images.push_back(img);
    }
    
    return images;
}

// Utility functions
double relu(double x) { return max(0.0, x); }
double relu_deriv(double x) { return x > 0 ? 1.0 : 0.0; }

vector<double> softmax(const vector<double>& input) {
    vector<double> output(input.size());
    double max_val = *max_element(input.begin(), input.end());
    double sum = 0;
    for (int i = 0; i < input.size(); i++) {
        output[i] = exp(input[i] - max_val);  // Subtract max for numerical stability
        sum += output[i];
    }
    for (int i = 0; i < input.size(); i++) {
        output[i] /= sum;
    }
    return output;
}

// CNN Class
class CNN {
private:
    // Layer 1
    vector<vector<vector<double>>> conv1_weights;  // 5x5 filters, 6 feature maps
    vector<double> conv1_bias;
    
    // Layer 2
    vector<vector<vector<double>>> conv2_weights;  // 5x5 filters, 16 feature maps
    vector<double> conv2_bias;
    
    // Fully connected layers
    vector<vector<double>> fc1_weights;
    vector<double> fc1_bias;
    vector<vector<double>> fc2_weights;  // Output layer
    vector<double> fc2_bias;

    const int INPUT_SIZE = 28;
    const int FILTER_SIZE = 5;
    const int CONV1_FILTERS = 6;
    const int CONV2_FILTERS = 16;
    const int POOL_SIZE = 2;
    const int FC1_SIZE = 120;
    const int OUTPUT_SIZE = 10;

    // Training parameters
    double learning_rate = 0.01;

public:
    CNN() {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> d(0, 0.1);

        // Initialize Conv1
        conv1_weights.resize(CONV1_FILTERS, vector<vector<double>>(FILTER_SIZE, vector<double>(FILTER_SIZE)));
        conv1_bias.resize(CONV1_FILTERS);
        for (int f = 0; f < CONV1_FILTERS; f++) {
            conv1_bias[f] = d(gen);
            for (int i = 0; i < FILTER_SIZE; i++) {
                for (int j = 0; j < FILTER_SIZE; j++) {
                    conv1_weights[f][i][j] = d(gen);
                }
            }
        }

        // Initialize Conv2
        conv2_weights.resize(CONV2_FILTERS, vector<vector<double>>(FILTER_SIZE, vector<double>(FILTER_SIZE)));
        conv2_bias.resize(CONV2_FILTERS);
        for (int f = 0; f < CONV2_FILTERS; f++) {
            conv2_bias[f] = d(gen);
            for (int i = 0; i < FILTER_SIZE; i++) {
                for (int j = 0; j < FILTER_SIZE; j++) {
                    conv2_weights[f][i][j] = d(gen);
                }
            }
        }

        // Initialize FC layers
        int conv2_out_size = (((INPUT_SIZE - FILTER_SIZE + 1) / POOL_SIZE) - FILTER_SIZE + 1) / POOL_SIZE;
        int fc1_input = CONV2_FILTERS * conv2_out_size * conv2_out_size;
        
        fc1_weights.resize(FC1_SIZE, vector<double>(fc1_input));
        fc1_bias.resize(FC1_SIZE);
        for (int i = 0; i < FC1_SIZE; i++) {
            fc1_bias[i] = d(gen);
            for (int j = 0; j < fc1_input; j++) {
                fc1_weights[i][j] = d(gen);
            }
        }

        fc2_weights.resize(OUTPUT_SIZE, vector<double>(FC1_SIZE));
        fc2_bias.resize(OUTPUT_SIZE);
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            fc2_bias[i] = d(gen);
            for (int j = 0; j < FC1_SIZE; j++) {
                fc2_weights[i][j] = d(gen);
            }
        }
    }

    vector<vector<vector<double>>> convolve(const vector<vector<double>>& input, 
                                          const vector<vector<vector<double>>>& weights, 
                                          const vector<double>& bias, int num_filters, 
                                          int input_size) {
        int output_size = input_size - FILTER_SIZE + 1;
        vector<vector<vector<double>>> output(num_filters, 
            vector<vector<double>>(output_size, vector<double>(output_size)));
        
        for (int f = 0; f < num_filters; f++) {
            for (int i = 0; i < output_size; i++) {
                for (int j = 0; j < output_size; j++) {
                    double sum = 0;
                    for (int m = 0; m < FILTER_SIZE; m++) {
                        for (int n = 0; n < FILTER_SIZE; n++) {
                            sum += input[i + m][j + n] * weights[f][m][n];
                        }
                    }
                    output[f][i][j] = relu(sum + bias[f]);
                }
            }
        }
        return output;
    }

    vector<vector<vector<double>>> maxPool(const vector<vector<vector<double>>>& input) {
        int input_size = input[0].size();
        int output_size = input_size / POOL_SIZE;
        vector<vector<vector<double>>> output(input.size(), 
            vector<vector<double>>(output_size, vector<double>(output_size)));
        
        for (int f = 0; f < input.size(); f++) {
            for (int i = 0; i < output_size; i++) {
                for (int j = 0; j < output_size; j++) {
                    double max_val = input[f][i*2][j*2];
                    for (int m = 0; m < POOL_SIZE; m++) {
                        for (int n = 0; n < POOL_SIZE; n++) {
                            max_val = max(max_val, input[f][i*2 + m][j*2 + n]);
                        }
                    }
                    output[f][i][j] = max_val;
                }
            }
        }
        return output;
    }

    struct ForwardPass {
        vector<vector<vector<double>>> conv1_out, pool1_out, conv2_out, pool2_out;
        vector<double> fc1_out, fc2_out, softmax_out;
    };

    ForwardPass forward(const vector<vector<double>>& input) {
        ForwardPass fp;
        
        // First conv layer
        fp.conv1_out = convolve(input, conv1_weights, conv1_bias, CONV1_FILTERS, INPUT_SIZE);
        fp.pool1_out = maxPool(fp.conv1_out);
        
        // Second conv layer
        int pool1_size = fp.pool1_out[0].size();
        fp.conv2_out = convolve(fp.pool1_out[0], conv2_weights, conv2_bias, CONV2_FILTERS, pool1_size);
        fp.pool2_out = maxPool(fp.conv2_out);

        // Flatten
        int pool2_size = fp.pool2_out[0].size();
        vector<double> flattened(CONV2_FILTERS * pool2_size * pool2_size);
        int idx = 0;
        for (int f = 0; f < CONV2_FILTERS; f++) {
            for (int i = 0; i < pool2_size; i++) {
                for (int j = 0; j < pool2_size; j++) {
                    flattened[idx++] = fp.pool2_out[f][i][j];
                }
            }
        }

        // FC1
        fp.fc1_out.resize(FC1_SIZE);
        for (int i = 0; i < FC1_SIZE; i++) {
            double sum = 0;
            for (int j = 0; j < flattened.size(); j++) {
                sum += flattened[j] * fc1_weights[i][j];
            }
            fp.fc1_out[i] = relu(sum + fc1_bias[i]);
        }

        // FC2 (output)
        fp.fc2_out.resize(OUTPUT_SIZE);
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            double sum = 0;
            for (int j = 0; j < FC1_SIZE; j++) {
                sum += fp.fc1_out[j] * fc2_weights[i][j];
            }
            fp.fc2_out[i] = sum + fc2_bias[i];
        }

        fp.softmax_out = softmax(fp.fc2_out);
        return fp;
    }

    void train(const vector<Image>& training_data, int epochs, int batch_size) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double total_loss = 0;
            int correct = 0;
            
            // Shuffle data
            vector<Image> shuffled_data = training_data;
            random_shuffle(shuffled_data.begin(), shuffled_data.end());

            for (int i = 0; i < training_data.size(); i += batch_size) {
                // Simple SGD
                Image& img = shuffled_data[i];
                ForwardPass fp = forward(img.data);

                // Compute loss (cross-entropy)
                vector<double> target(OUTPUT_SIZE, 0);
                target[img.label] = 1.0;
                double loss = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    loss -= target[j] * log(fp.softmax_out[j] + 1e-10);
                }
                total_loss += loss;

                // Backpropagation (simplified)
                vector<double> output_error(OUTPUT_SIZE);
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    output_error[j] = fp.softmax_out[j] - target[j];
                }

                // Update FC2
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    fc2_bias[j] -= learning_rate * output_error[j];
                    for (int k = 0; k < FC1_SIZE; k++) {
                        fc2_weights[j][k] -= learning_rate * output_error[j] * fp.fc1_out[k];
                    }
                }

                // Add more backprop layers as needed...

                // Track accuracy
                int predicted = max_element(fp.softmax_out.begin(), fp.softmax_out.end()) - fp.softmax_out.begin();
                if (predicted == img.label) correct++;
            }

            cout << "Epoch " << epoch + 1 << ": Loss = " << total_loss / training_data.size()
                 << ", Accuracy = " << (correct * 100.0 / training_data.size()) << "%" << endl;
        }
    }

    double evaluate(const vector<Image>& test_data) {
        int correct = 0;
        for (const auto& img : test_data) {
            ForwardPass fp = forward(img.data);
            int predicted = max_element(fp.softmax_out.begin(), fp.softmax_out.end()) - fp.softmax_out.begin();
            if (predicted == img.label) correct++;
        }
        return (correct * 100.0) / test_data.size();
    }

    void printParameters() {
        cout << "\nFinal CNN Parameters:\n";
        
        // Conv1 Filters and Biases
        cout << "\nConvolutional Layer 1:\n";
        cout << "Filters (" << CONV1_FILTERS << " x " << FILTER_SIZE << " x " << FILTER_SIZE << "):\n";
        for (int f = 0; f < CONV1_FILTERS; f++) {
            cout << "Filter " << f + 1 << ":\n";
            for (int i = 0; i < FILTER_SIZE; i++) {
                for (int j = 0; j < FILTER_SIZE; j++) {
                    cout << conv1_weights[f][i][j] << " ";
                }
                cout << endl;
            }
            cout << "Bias: " << conv1_bias[f] << "\n\n";
        }

        // Conv2 Filters and Biases
        cout << "\nConvolutional Layer 2:\n";
        cout << "Filters (" << CONV2_FILTERS << " x " << FILTER_SIZE << " x " << FILTER_SIZE << "):\n";
        for (int f = 0; f < min(2, CONV2_FILTERS); f++) {  // Print first 2 filters only (to avoid too much output)
            cout << "Filter " << f + 1 << ":\n";
            for (int i = 0; i < FILTER_SIZE; i++) {
                for (int j = 0; j < FILTER_SIZE; j++) {
                    cout << conv2_weights[f][i][j] << " ";
                }
                cout << endl;
            }
            cout << "Bias: " << conv2_bias[f] << "\n\n";
        }
        if (CONV2_FILTERS > 2) cout << "... (remaining " << CONV2_FILTERS-2 << " filters not shown)\n";

        // FC1 Weights and Biases
        cout << "\nFully Connected Layer 1:\n";
        cout << "Weights (" << FC1_SIZE << " x " << fc1_weights[0].size() << "), showing first neuron:\n";
        for (int j = 0; j < min(5, (int)fc1_weights[0].size()); j++) {
            cout << fc1_weights[0][j] << " ";
        }
        cout << "...\n";
        cout << "Biases (" << FC1_SIZE << "), showing first 5:\n";
        for (int i = 0; i < min(5, FC1_SIZE); i++) {
            cout << fc1_bias[i] << " ";
        }
        cout << "...\n";

        // FC2 Weights and Biases
        cout << "\nFully Connected Layer 2 (Output):\n";
        cout << "Weights (" << OUTPUT_SIZE << " x " << FC1_SIZE << "), showing first class:\n";
        for (int j = 0; j < min(5, FC1_SIZE); j++) {
            cout << fc2_weights[0][j] << " ";
        }
        cout << "...\n";
        cout << "Biases (" << OUTPUT_SIZE << "):\n";
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            cout << fc2_bias[i] << " ";
        }
        cout << endl;
    }
};

// MNIST loading function remains the same as previous version

int main() {
    // Load data
    vector<Image> train_data = readMNISTImages("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 60000);
    vector<Image> test_data = readMNISTImages("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 10000);

    // Create and train CNN
    CNN cnn;
    cout << "Training started..." << endl;
    cnn.train(train_data, 5, 1);  // 5 epochs, batch size 1
    
    // Evaluate
    double accuracy = cnn.evaluate(test_data);
    cout << "Test accuracy: " << accuracy << "%" << endl;

    cnn.printParameters();

    return 0;
}