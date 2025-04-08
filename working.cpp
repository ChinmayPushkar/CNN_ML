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
    
    // // Layer 2
    // vector<vector<vector<double>>> conv2_weights;  // 5x5 filters, 16 feature maps
    // vector<double> conv2_bias;
    
    // Fully connected layers
    vector<vector<double>> fc1_weights;
    vector<double> fc1_bias;
    vector<vector<double>> fc2_weights;  // Output layer
    vector<double> fc2_bias;

    const int INPUT_SIZE = 28;
    const int FILTER_SIZE = 3;
    const int CONV1_FILTERS = 6;
    const int POOL_SIZE = 2;
    const int FC1_SIZE = 120;
    const int OUTPUT_SIZE = 10;

    // Training parameters
    double learning_rate = 0.01;

	// Gradients storage
	vector<vector<vector<double>>> conv1_weight_grads;
    vector<double> conv1_bias_grads;
    vector<vector<double>> fc1_weight_grads;
    vector<double> fc1_bias_grads;
    vector<vector<double>> fc2_weight_grads;
    vector<double> fc2_bias_grads;

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

		conv1_weight_grads.resize(CONV1_FILTERS, vector<vector<double>>(FILTER_SIZE, vector<double>(FILTER_SIZE, 0)));
        conv1_bias_grads.resize(CONV1_FILTERS, 0);
        
        int conv1_out_size = ((INPUT_SIZE - FILTER_SIZE + 1) / POOL_SIZE);
        int fc1_input = CONV1_FILTERS * conv1_out_size * conv1_out_size;
        
        fc1_weight_grads.resize(FC1_SIZE, vector<double>(fc1_input, 0));
        fc1_bias_grads.resize(FC1_SIZE, 0);
        fc2_weight_grads.resize(OUTPUT_SIZE, vector<double>(FC1_SIZE, 0));
        fc2_bias_grads.resize(OUTPUT_SIZE, 0);
        
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
        vector<vector<vector<double>>> conv1_out, pool1_out;
        vector<vector<vector<int>>> pool1_max_indices;  // Store max pool indices
        vector<double> flatten_out, fc1_out, fc2_out, softmax_out;
    };

	ForwardPass forward(const vector<vector<double>>& input) {
        ForwardPass fp;
        
        fp.conv1_out = convolve(input, conv1_weights, conv1_bias, CONV1_FILTERS, INPUT_SIZE);
        
        // Modified maxPool to store indices
        int input_size = fp.conv1_out[0].size();
        int output_size = input_size / POOL_SIZE;
        fp.pool1_out.resize(CONV1_FILTERS, vector<vector<double>>(output_size, vector<double>(output_size)));
        fp.pool1_max_indices.resize(CONV1_FILTERS, vector<vector<int>>(output_size, vector<int>(output_size)));
        
        for (int f = 0; f < CONV1_FILTERS; f++) {
            for (int i = 0; i < output_size; i++) {
                for (int j = 0; j < output_size; j++) {
                    double max_val = fp.conv1_out[f][i*2][j*2];
                    int max_idx = 0;
                    for (int m = 0; m < POOL_SIZE; m++) {
                        for (int n = 0; n < POOL_SIZE; n++) {
                            double val = fp.conv1_out[f][i*2 + m][j*2 + n];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = m * POOL_SIZE + n;
                            }
                        }
                    }
                    fp.pool1_out[f][i][j] = max_val;
                    fp.pool1_max_indices[f][i][j] = max_idx;
                }
            }
        }

        // Flatten
        int pool1_size = fp.pool1_out[0].size();
		// cout<<pool1_size<<endl;
        fp.flatten_out.resize(CONV1_FILTERS * pool1_size * pool1_size);
        int idx = 0;
        for (int f = 0; f < CONV1_FILTERS; f++) {
            for (int i = 0; i < pool1_size; i++) {
                for (int j = 0; j < pool1_size; j++) {
                    fp.flatten_out[idx++] = fp.pool1_out[f][i][j];
                }
            }
        }

        // FC1
        fp.fc1_out.resize(FC1_SIZE);
        for (int i = 0; i < FC1_SIZE; i++) {
            double sum = 0;
            for (int j = 0; j < fp.flatten_out.size(); j++) {
                sum += fp.flatten_out[j] * fc1_weights[i][j];
            }
            fp.fc1_out[i] = relu(sum + fc1_bias[i]);
        }

        // FC2
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

	void backward(const vector<vector<double>>& input, const ForwardPass& fp, 
                 const vector<double>& target) {
        // Reset gradients
        fill(conv1_bias_grads.begin(), conv1_bias_grads.end(), 0);
        fill(fc1_bias_grads.begin(), fc1_bias_grads.end(), 0);
        fill(fc2_bias_grads.begin(), fc2_bias_grads.end(), 0);
        
        for (auto& filter : conv1_weight_grads)
            for (auto& row : filter)
                fill(row.begin(), row.end(), 0);
        for (auto& row : fc1_weight_grads)
            fill(row.begin(), row.end(), 0);
        for (auto& row : fc2_weight_grads)
            fill(row.begin(), row.end(), 0);

        // 1. Output layer (FC2) gradients
        vector<double> fc2_delta(OUTPUT_SIZE);
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            fc2_delta[i] = fp.softmax_out[i] - target[i];
            fc2_bias_grads[i] = fc2_delta[i];
            for (int j = 0; j < FC1_SIZE; j++) {
                fc2_weight_grads[i][j] = fc2_delta[i] * fp.fc1_out[j];
            }
        }

        // 2. FC1 gradients
        vector<double> fc1_delta(FC1_SIZE);
        for (int i = 0; i < FC1_SIZE; i++) {
            double sum = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                sum += fc2_delta[j] * fc2_weights[j][i];
            }
            fc1_delta[i] = sum * relu_deriv(fp.fc1_out[i]);
            fc1_bias_grads[i] = fc1_delta[i];
            for (int j = 0; j < fp.flatten_out.size(); j++) {
                fc1_weight_grads[i][j] = fc1_delta[i] * fp.flatten_out[j];
            }
        }

        // 3. Pool1 and Conv1 gradients
        int pool_size = fp.pool1_out[0].size();
        vector<vector<vector<double>>> pool1_delta(CONV1_FILTERS, 
            vector<vector<double>>(pool_size, vector<double>(pool_size)));
        
        // Unflatten fc1_delta
        vector<vector<vector<double>>> flatten_delta(CONV1_FILTERS, 
            vector<vector<double>>(pool_size, vector<double>(pool_size)));
        int idx = 0;
        for (int f = 0; f < CONV1_FILTERS; f++) {
            for (int i = 0; i < pool_size; i++) {
                for (int j = 0; j < pool_size; j++) {
                    double sum = 0;
                    for (int k = 0; k < FC1_SIZE; k++) {
                        sum += fc1_delta[k] * fc1_weights[k][idx];
                    }
                    flatten_delta[f][i][j] = sum;
                    idx++;
                }
            }
        }

        // MaxPool backward
        int conv1_size = fp.conv1_out[0].size();
        vector<vector<vector<double>>> conv1_delta(CONV1_FILTERS, 
            vector<vector<double>>(conv1_size, vector<double>(conv1_size, 0)));
            
        for (int f = 0; f < CONV1_FILTERS; f++) {
            for (int i = 0; i < pool_size; i++) {
                for (int j = 0; j < pool_size; j++) {
                    int max_idx = fp.pool1_max_indices[f][i][j];
                    int m = max_idx / POOL_SIZE;
                    int n = max_idx % POOL_SIZE;
                    conv1_delta[f][i*2 + m][j*2 + n] = flatten_delta[f][i][j];
                }
            }
        }

        // Conv1 backward
        for (int f = 0; f < CONV1_FILTERS; f++) {
            for (int i = 0; i < conv1_size; i++) {
                for (int j = 0; j < conv1_size; j++) {
                    if (fp.conv1_out[f][i][j] > 0) {  // ReLU derivative
                        double delta = conv1_delta[f][i][j];
                        conv1_bias_grads[f] += delta;
                        for (int m = 0; m < FILTER_SIZE; m++) {
                            for (int n = 0; n < FILTER_SIZE; n++) {
                                if (i+m < INPUT_SIZE && j+n < INPUT_SIZE) {
                                    conv1_weight_grads[f][m][n] += delta * input[i+m][j+n];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Update weights and biases
        for (int f = 0; f < CONV1_FILTERS; f++) {
            conv1_bias[f] -= learning_rate * conv1_bias_grads[f];
            for (int i = 0; i < FILTER_SIZE; i++) {
                for (int j = 0; j < FILTER_SIZE; j++) {
                    conv1_weights[f][i][j] -= learning_rate * conv1_weight_grads[f][i][j];
                }
            }
        }

        for (int i = 0; i < FC1_SIZE; i++) {
            fc1_bias[i] -= learning_rate * fc1_bias_grads[i];
            for (int j = 0; j < fp.flatten_out.size(); j++) {
                fc1_weights[i][j] -= learning_rate * fc1_weight_grads[i][j];
            }
        }

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            fc2_bias[i] -= learning_rate * fc2_bias_grads[i];
            for (int j = 0; j < FC1_SIZE;j++) {
                fc2_weights[i][j] -= learning_rate * fc2_weight_grads[i][j];
            }
        }
    }

    void train(const vector<Image>& training_data, int epochs, int batch_size=1) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double total_loss = 0;
            int correct = 0;
            vector<Image> shuffled_data = training_data;
            static std::random_device rd;
			static std::mt19937 g(rd());
			std::shuffle(shuffled_data.begin(), shuffled_data.end(), g);


            for (int i = 0; i < training_data.size(); i += batch_size) {
                Image& img = shuffled_data[i];
                ForwardPass fp = forward(img.data);

                vector<double> target(OUTPUT_SIZE, 0);
                target[img.label] = 1.0;
                
                // Compute loss
                double loss = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    loss -= target[j] * log(fp.softmax_out[j] + 1e-10);
                }
                total_loss += loss;

                // Backward pass and weight update
                backward(img.data, fp, target);

                int predicted = max_element(fp.softmax_out.begin(), fp.softmax_out.end()) - fp.softmax_out.begin();
                if (predicted == img.label) correct++;
            }

            cout << "Epoch " << epoch + 1 << ": Loss = " << total_loss / training_data.size()
                 << ", Accuracy = " << (correct * 100.0 / training_data.size()) << "%" << endl;
        }
    }

    double evaluate(const vector<Image>& test_data, vector<vector<int>>& confusion_matrix) {
		int correct = 0;
		
		// Resize and initialize confusion matrix to zeros
		confusion_matrix.resize(10, vector<int>(10, 0));
		
		for (const auto& img : test_data) {
			ForwardPass fp = forward(img.data);
			int predicted = max_element(fp.softmax_out.begin(), fp.softmax_out.end()) - fp.softmax_out.begin();
			
			// Update confusion matrix: actual label (row) vs predicted label (column)
			confusion_matrix[img.label][predicted]++;
			
			if (predicted == img.label) correct++;
		}
		
		return (correct * 100.0) / test_data.size();
	}
	void printConfusionMatrix(const vector<vector<int>>& confusion_matrix) {
		cout << "\nConfusion Matrix (rows: actual, columns: predicted):\n";
		cout << "     ";  // Align header
		for (int i = 0; i < 10; i++) {
			cout << setw(4) << i;
		}
		cout << "\n   +----------------------------------------+\n";
		
		for (int i = 0; i < 10; i++) {
			cout << setw(2) << i << " |";
			for (int j = 0; j < 10; j++) {
				cout << setw(4) << confusion_matrix[i][j];
			}
			cout << " |\n";
		}
		cout << "   +----------------------------------------+\n";
		
		// Print per-class accuracy
		cout << "\nPer-class accuracy:\n";
		for (int i = 0; i < 10; i++) {
			int true_positives = confusion_matrix[i][i];
			int total = 0;
			for (int j = 0; j < 10; j++) {
				total += confusion_matrix[i][j];
			}
			double class_accuracy = total > 0 ? (true_positives * 100.0 / total) : 0.0;
			cout << "Class " << i << ": " << fixed << setprecision(2) << class_accuracy << "%\n";
		}
	}

    void printParameters() {
        cout << "\nFinal CNN Parameters:\n";
        
        // Conv1 Filters and Biases
        cout << "\nConvolutional Layer 1:\n";
		cout<<CONV1_FILTERS<<endl;
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

int main() {
    // Load data
    vector<Image> train_data = readMNISTImages("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 60000);
    vector<Image> test_data = readMNISTImages("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 10000);

    // Create and train CNN
    CNN cnn;
    cout << "Training started..." << endl;
    cnn.train(train_data, 5, 1);  // 5 epochs, batch size 1
    
    // Evaluate with confusion matrix
    vector<vector<int>> confusion_matrix;
    double accuracy = cnn.evaluate(test_data, confusion_matrix);
    cout << "Test accuracy: " << accuracy << "%" << endl;

    // Print confusion matrix
    cnn.printConfusionMatrix(confusion_matrix);

    // cnn.printParameters();  // Uncomment if you still want this

    return 0;
}